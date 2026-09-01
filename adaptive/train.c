/* train.c - entrenador del LLM adaptativo: backprop + Adam + gate PPL
 * (se construye por bloques: este es el bloque 1/5: includes + CLI)
 */
#include "ad_train.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef USE_GPU
#include "ad_gpu.h"
static int OPT_GPU = 0;
#endif

/* ---- opciones globales ---- */
static int    OPT_STEPS = 2000;
static int    OPT_T     = 96;
static float  OPT_LR    = 3e-4f;
static int    OPT_EVAL  = 100;
static int    OPT_VOCAB = 256;      /* 2048 => modo BPE */
static int    OPT_BATCH = 8;

static uint8_t *read_file(const char *path, size_t *len);
/* lee corpus BAR1 (u32 ids) o bytes crudos (id=byte); malloc result, n por puntero */
static int *read_tokens(const char *path, size_t *n) {
    size_t len = 0;
    uint8_t *raw = read_file(path, &len);
    if (!raw) return NULL;
    int *toks;
    if (len >= 8 && raw[0]=='B' && raw[1]=='A' && raw[2]=='R' && raw[3]=='1') {
        uint32_t n4;
        memcpy(&n4, raw + 4, 4);
        toks = (int *)malloc((size_t)n4 * sizeof(int));
        memcpy(toks, raw + 8, (size_t)n4 * sizeof(int));
        free(raw);
        *n = n4;
    } else {
        toks = (int *)malloc(len * sizeof(int));
        for (size_t i = 0; i < len; i++) toks[i] = raw[i];
        free(raw);
        *n = len;
    }
    return toks;
}

/* ==================================================================
 *  FORWARD + BACKWARD de una ventana de T bytes
 *  targets = bytes[t+1]; usa buffers bX/bL1/... ya allocados
 *  acumula gradientes en GRAD (arena espejo de M.w)
 *  devuelve CE media por prediccion (T-1)
 * ================================================================== */
static float win_fwd_bwd(AdModel *m, const int *bytes, int T, int do_bwd);

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "OOM %zu\n", n); exit(1); }
    return p;
}

static uint8_t *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return NULL; }
    uint8_t *u = (uint8_t *)malloc((size_t)sz);
    if (!u) { fclose(f); return NULL; }
    if (fread(u, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(u); return NULL; }
    fclose(f);
    *len = (size_t)sz;
    return u;
}

/* contexto global del entrenamiento (main usa esto; WASM lo setea directo) */
static AdModel M;               /* modelo en entrenamiento */
static float  *GRAD, *ADM, *ADV;/* gradientes + moments Adam */
static int *g_train = NULL, *g_val = NULL;
static size_t   g_train_len = 0, g_val_len = 0;
static int      g_adam_t = 0;
static size_t   g_nfloats = 0;
static int      g_opt_muon = 0;          /* --opt muon (PuRo 3.3) */
static float   *AMM = NULL;              /* momentum Muon (espejo de arena) */
static size_t   AMM_cap = 0;

static float win_fwd_bwd(AdModel *m, const int *bts, int T, int do_bwd);

/* gradiente de pesos de un linear: dW[r,c] += dy[r] * x[c] */
static void gWadd(float *dW, const float *dy, const float *x, int R, int C) {
    for (int r = 0; r < R; r++) {
        float *dw = dW + (size_t)r * C;
        float yv = dy[r];
        for (int c = 0; c < C; c++) dw[c] += yv * x[c];
    }
}

/* layernorm forward guardando m,s para backward */
static void ln_fwd(float *out, const float *x, const float *g, const float *b,
                   int n, float *m_o, float *s_o) {
    float m = 0.f;
    for (int i = 0; i < n; i++) m += x[i];
    m /= n;
    float v = 0.f;
    for (int i = 0; i < n; i++) { float d = x[i] - m; v += d * d; }
    v /= n;
    float s = 1.0f / sqrtf(v + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = (x[i] - m) * s * g[i] + b[i];
    *m_o = m; *s_o = s;
}

/* ---- buffers de activaciones (se allocan al conocer dim/T) ---- */
static float *bX, *bL1, *bQKV, *bAO, *bPR, *bL2, *bF1, *bFA, *bLF, *bLG;
static float *bDX, *bDP, *bSCR, *bATW;
static float *bDLG;             /* dlogits [T*V] */
static int   bT = 0;            /* T real de las ventanas */
static int   bD = 0, bH = 0;    /* dim, hidden */
static int   bHD = 0, bNH = 0, bL = 0;

#define XROW(l, t) (bX + (((size_t)(l) * (size_t)bT + (size_t)(t))) * bD)

static int bufs_alloc(void) {
    size_t d = (size_t)bD, hid = (size_t)bH, T = (size_t)bT;
    size_t L = (size_t)bL;
    bX   = (float *)xmalloc((L + 1) * T * d * sizeof(float));
    bL1  = (float *)xmalloc(L * T * d * sizeof(float));
    bQKV = (float *)xmalloc(L * T * 3 * d * sizeof(float));
    bAO  = (float *)xmalloc(L * T * d * sizeof(float));
    bPR  = (float *)xmalloc(L * T * d * sizeof(float));
    bL2  = (float *)xmalloc(L * T * d * sizeof(float));
    bF1  = (float *)xmalloc(L * T * hid * sizeof(float));
    bFA  = (float *)xmalloc(L * T * hid * sizeof(float));
    bLF  = (float *)xmalloc(T * d * sizeof(float));
    bLG  = (float *)xmalloc(T * AD_VOCAB * sizeof(float));
    bDLG = (float *)xmalloc(T * AD_VOCAB * sizeof(float));
    bDX  = (float *)xmalloc(T * d * sizeof(float));
    bDP  = (float *)xmalloc(T * d * sizeof(float));
    bSCR = (float *)xmalloc(T * sizeof(float));
    bATW = (float *)xmalloc((size_t)bL * bNH * T * T * sizeof(float));
    if (!bX || !bL1 || !bQKV || !bAO || !bPR || !bL2 || !bF1 || !bFA
        || !bLF || !bLG || !bDX || !bDP || !bSCR || !bATW) return -1;
    return 0;
}

/* ---- setup: buffers + gradientes + Adam (tras cargar M) ---- */
static int train_setup(void) {
    bT = OPT_T; bD = M.cfg.dim; bH = M.cfg.hidden;
    bHD = M.cfg.head_dim; bNH = M.cfg.n_heads; bL = M.cfg.n_layers;
    if (bufs_alloc()) return -1;
    g_nfloats = ad_total_floats(&M.cfg);
    GRAD = (float *)calloc(g_nfloats, sizeof(float));
    ADM  = (float *)calloc(g_nfloats, sizeof(float));
    ADV  = (float *)calloc(g_nfloats, sizeof(float));
    if (g_opt_muon) AMM = (float *)calloc(g_nfloats, sizeof(float));
    if (!GRAD || !ADM || !ADV || (g_opt_muon && !AMM)) return -2;
    return 0;
}

/* ---- eval: PPL media sobre el corpus val en ventanas aleatorias ---- */
static double eval_ppl_val(const int *val, size_t val_len, int n_win) {
    if (val_len < (size_t)bT + 2) return 0.0;
    double tot = 0.0;
    for (int w = 0; w < n_win; w++) {
        size_t st = (size_t)(ad_randf()
                    * (double)(val_len - (size_t)bT - 1));
        tot += win_fwd_bwd(&M, val + st, bT, 0);
    }
    return exp(tot / n_win);
}

/* ---- chunk: steps x batch con Adam + cosine schedule + warmup + clipping
 * (schedule segun Karpeles ch.3: LR alto temprano, decae suave al final;
 *  recorte de gradiente global para evitar explosiones sin supervision)
 * total_steps: horizon de steps para el cosine (si 0 => sin schedule) ---- */
static double train_chunk(int steps, int batch, float lr, int total_steps) {
    double total = 0.0;
    for (int step = 0; step < steps; step++) {
        double avg = 0.0;
        float grad_norm = 0.f;
        for (int bi = 0; bi < batch; bi++) {
            size_t st = (size_t)(ad_randf()
                        * (double)(g_train_len - (size_t)bT - 1));
            avg += win_fwd_bwd(&M, g_train + st, bT, 1);
        }
        /* LR efectivo: WSD (Warmup-Stable-Decay) o cosine
         * WSD (PuRo-2B 3.3.2): warmup 10%, estable al LR base hasta 80%,
         * decae lineal a ~0 el 20% final. Ventana: resumir en cualquier
         * punto estable sin valle de cosine */
        float lr_now = lr;
        if (total_steps > 0) {
            int t = g_adam_t, T = total_steps;
            int warm = T / 10, decay = T / 5;   /* 10% warmup, 20% decay */
            if (t < warm) {
                lr_now = lr * (float)(t + 1) / (float)warm;
            } else if (t >= T - decay) {
                float prog = (float)(t - (T - decay)) / (float)decay;
                if (prog > 1.f) prog = 1.f;
                lr_now = lr * (1.0f - 0.95f * prog);   /* decae a 5% del base */
            }
            /* entre warm y T-decay: LR estable = lr (WSD) */
        }
        /* clipping: norma L2 global del gradiente (recorta si > 1.0) */
        for (size_t i = 0; i < g_nfloats; i++)
            grad_norm += GRAD[i] * GRAD[i];
        grad_norm = sqrtf(grad_norm / (float)batch);
        float scale = 1.0f;
        if (grad_norm > 1.0f) scale = 1.0f / grad_norm;

        g_adam_t++;
        float bc1 = 1.0f - powf(0.9f, (float)g_adam_t);
        float bc2 = 1.0f - powf(0.999f, (float)g_adam_t);
#ifdef USE_GPU
        if (OPT_GPU) {
            /* GPU path: Adam runs on device, skip CPU update */
            ad_gpu_adam_step(lr_now, 0.9f, 0.999f, 1e-8f, 0.0f);
            total += avg / batch;
            continue;
        }
#endif
        if (!g_opt_muon) {
            /* AdamW clasico para todos los tensores */
            for (size_t i = 0; i < g_nfloats; i++) {
                float gi = (GRAD[i] / (float)batch) * scale;
                ADM[i] = 0.9f * ADM[i] + 0.1f * gi;
                ADV[i] = 0.999f * ADV[i] + 0.001f * gi * gi;
                M.w[i] -= lr_now * (ADM[i] / bc1)
                        / (sqrtf(ADV[i] / bc2) + 1e-8f);
                GRAD[i] = 0.f;
            }
        } else {
            /* MuonH (PuRo 3.3): matrices 2D por bloque = MuonH,
             * emb/pos/ln/biases/head = AdamW (LR base); matrices con
             * LR = 10x base. Se aplica por tensor con el layout. */
            size_t V = (size_t)(M.cfg.vocab > 0 ? M.cfg.vocab : 256);
            size_t d = (size_t)M.cfg.dim, hid = (size_t)M.cfg.hidden;
            /* 1) tensores AdamW globales: tok_emb y head (grandes/V-vocab) */
            for (size_t i = 0; i < V * d; i++) {
                size_t gb = M.lay.tok_emb + i;
                float gi = (GRAD[gb] / (float)batch) * scale;
                ADM[gb] = 0.9f * ADM[gb] + 0.1f * gi;
                ADV[gb] = 0.999f * ADV[gb] + 0.001f * gi * gi;
                M.w[gb] -= lr_now * (ADM[gb] / bc1)
                        / (sqrtf(ADV[gb] / bc2) + 1e-8f);
                GRAD[gb] = 0.f;
                gb = M.lay.w_head + i;   /* head igual */
                gi = (GRAD[gb] / (float)batch) * scale;
                ADM[gb] = 0.9f * ADM[gb] + 0.1f * gi;
                ADV[gb] = 0.999f * ADV[gb] + 0.001f * gi * gi;
                M.w[gb] -= lr_now * (ADM[gb] / bc1)
                        / (sqrtf(ADV[gb] / bc2) + 1e-8f);
                GRAD[gb] = 0.f;
            }
            /* pos_emb: AdamW (es 1D-like vectorial: [seq x dim]) */
            for (size_t i = 0; i < (size_t)M.cfg.max_seq * d; i++) {
                size_t gb = M.lay.pos_emb + i;
                float gi = (GRAD[gb] / (float)batch) * scale;
                ADM[gb] = 0.9f * ADM[gb] + 0.1f * gi;
                ADV[gb] = 0.999f * ADV[gb] + 0.001f * gi * gi;
                M.w[gb] -= lr_now * (ADM[gb] / bc1)
                        / (sqrtf(ADV[gb] / bc2) + 1e-8f);
                GRAD[gb] = 0.f;
            }
            /* LNs y tail: AdamW */
            size_t lnfg_lo = M.lay.lnf_g, ln_end = M.lay.b_head;
            for (size_t i = lnfg_lo; i < ln_end && i < g_nfloats; i++) {
                float gi = (GRAD[i] / (float)batch) * scale;
                ADM[i] = 0.9f * ADM[i] + 0.1f * gi;
                ADV[i] = 0.999f * ADV[i] + 0.001f * gi * gi;
                M.w[i] -= lr_now * (ADM[i] / bc1)
                        / (sqrtf(ADV[i] / bc2) + 1e-8f);
                GRAD[i] = 0.f;
            }
            /* por capa: MuonH sobre Wqkv Wproj W1 W2; resto AdamW */
            float lrH = lr_now * 10.0f;
            size_t dim = d;
            for (int l = 0; l < M.cfg.n_layers; l++) {
                size_t lb = M.lay.layers + (size_t)l * M.lay.per_layer;
                /* adam: ln1g ln1b bqkv bproj ln2g ln2b b1 b2 */
                size_t adam_offs[] = { lb, lb + (size_t)dim,
                    lb + 2*(size_t)dim + (size_t)dim*3*dim,
                    lb + 2*(size_t)dim + (size_t)dim*3*dim + 3*(size_t)dim + (size_t)dim*(size_t)dim,
                };
                for (int oi = 0; oi < 4; oi++) {
                    size_t gb = adam_offs[oi];
                    size_t len = (oi == 0 || oi == 1) ? (size_t)dim
                               : (oi == 2) ? 3*(size_t)dim : (size_t)dim;
                    for (size_t i = 0; i < len; i++) {
                        size_t idx = gb + i >= g_nfloats ? 0 : gb + i;
                        float gi = (GRAD[(int)idx] / (float)batch) * scale;
                        ADM[idx] = 0.9f*ADM[idx] + 0.1f*gi;
                        ADV[idx] = 0.999f*ADV[idx] + 0.001f*gi*gi;
                        M.w[idx] -= lr_now * (ADM[idx]/bc1)
                                / (sqrtf(ADV[idx]/bc2) + 1e-8f);
                        GRAD[idx] = 0.f;
                    }
                }
                /* MuonH en las 4 matrices del bloque (con momentum dedicados
                 * en buf_adam[matriz] via ADM si existe el espejo) */
                /* Wqkv [dim x 3dim] */
                ad_muon_apply(M.w + lb + 2*(size_t)dim,
                              GRAD + lb + 2*(size_t)dim,
                              AMM + lb + 2*(size_t)dim,
                              dim, 3*dim, lrH);
                memset(GRAD + lb + 2*(size_t)dim, 0,
                       (size_t)dim*3*dim*sizeof(float));
                /* Wproj [dim x dim]*/
                size_t wo = lb + 2*(size_t)dim + (size_t)dim*3*dim + 3*(size_t)dim;
                ad_muon_apply(M.w + wo, GRAD + wo, AMM + wo, dim, dim, lrH);
                memset(GRAD + wo, 0, (size_t)dim*(size_t)dim*sizeof(float));
                /* W1 [dim x hid] */
                size_t w1o = wo + (size_t)dim*(size_t)dim + (size_t)dim;
                ad_muon_apply(M.w + w1o, GRAD + w1o, AMM + w1o, dim, (int)hid, lrH);
                memset(GRAD + w1o, 0, (size_t)dim*(size_t)hid*sizeof(float));
                /* W2 [hid x dim] */
                size_t w2o = w1o + (size_t)dim*(size_t)hid + (size_t)hid;
                ad_muon_apply(M.w + w2o, GRAD + w2o, AMM + w2o, (int)hid, dim, lrH);
                memset(GRAD + w2o, 0, (size_t)hid*(size_t)dim*sizeof(float));
            }
        }
        total += avg / batch;
    }
    return total / steps;
}

int main(int argc, char **argv) {
    const char *model_in = NULL, *data_path = NULL, *val_path = NULL;
    const char *out_path = "model.adm";
    int fresh_d = 0, fresh_h = 0, fresh_l = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--fresh") && i + 1 < argc) {
            sscanf(argv[++i], "%d,%d,%d", &fresh_d, &fresh_h, &fresh_l);
        } else if (!strcmp(argv[i], "--model") && i + 1 < argc) {
            model_in = argv[++i];
        } else if (!strcmp(argv[i], "--out") && i + 1 < argc) {
            out_path = argv[++i];
        } else if (!strcmp(argv[i], "--data") && i + 1 < argc) {
            data_path = argv[++i];
        } else if (!strcmp(argv[i], "--val") && i + 1 < argc) {
            val_path = argv[++i];
        } else if (!strcmp(argv[i], "--steps") && i + 1 < argc) {
            OPT_STEPS = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) {
            OPT_T = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--lr") && i + 1 < argc) {
            OPT_LR = (float)atof(argv[++i]);
        } else if (!strcmp(argv[i], "--batch") && i + 1 < argc) {
            OPT_BATCH = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--eval") && i + 1 < argc) {
            OPT_EVAL = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--vocab") && i + 1 < argc) {
            OPT_VOCAB = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--opt") && i + 1 < argc) {
            if (!strcmp(argv[++i], "muon")) g_opt_muon = 1;
#ifdef USE_GPU
        } else if (!strcmp(argv[i], "--gpu")) {
            OPT_GPU = 1;
#endif
        }
    }
    if (!data_path || (!fresh_d && !model_in)) {
        fprintf(stderr,
            "uso: train --data corpus.bin (--fresh dim,H,L | --model m.adm)"
            " [--out model.adm] [--steps N] [--batch B] [--ctx T] [--lr F]"
            " [--vocab 2048] --val val.bin\n");
        return 1;
    }

    /* ---- cargar o crear modelo ---- */
    srand((unsigned)time(NULL));
    if (fresh_d > 0) {
        if (ad_init_fresh_v(&M, fresh_d, fresh_h, fresh_l, OPT_T * 2, OPT_VOCAB)) {
            fprintf(stderr, "no se pudo crear modelo fresh\n");
            return 2;
        }
        printf("modelo fresco: dim=%d L=%d H=%d vocab=%d\n",
               M.cfg.dim, fresh_h, fresh_l, M.cfg.vocab);
    } else {
        int r = ad_load(&M, model_in);
        if (r) { fprintf(stderr, "load %s -> %d\n", model_in, r); return 2; }
        printf("modelo cargado: steps=%u vocab=%d\n",
               M.cfg.train_steps, M.cfg.vocab);
    }

    /* ---- corpus (a globals; tambien los consume train_chunk) ---- */
    g_train = read_tokens(data_path, &g_train_len);
    g_val = val_path ? read_tokens(val_path, &g_val_len) : NULL;
    /* BARI corpus (ids u32): convierte a in-place bytes (id<256: byte; id>=256
     * expande a su secuencia de bytes via vocab del corpus? no - byte-level
     * fallback: los ids >=256 NO caben en byte => modo ids no soportado aqui */
    if (!g_train || g_train_len < (size_t)OPT_T + 2) {
        fprintf(stderr, "corpus %s vacio o pequeno\n", data_path);
        return 3;
    }
    if (g_val && g_val_len < (size_t)OPT_T + 2) { free(g_val); g_val = NULL; }

    /* ---- buffers + gradientes + Adam ---- */
    if (train_setup()) { fprintf(stderr, "OOM setup\n"); return 3; }

#ifdef USE_GPU
    /* ---- GPU init ---- */
    if (OPT_GPU) {
        if (ad_gpu_init(&M)) { fprintf(stderr, "GPU init failed, falling back to CPU\n"); OPT_GPU = 0; }
        else printf("GPU: modelo en device 0\n");
    }
#endif

    printf("b2: modelo+corpus+buffers OK (train=%zu val=%zu)\n",
           g_train_len, g_val ? g_val_len : 0);

    /* ================== BUCLE DE ENTRENAMIENTO ================== */
    double ppl0 = 0.0, ppl1 = 0.0;
    if (g_val) {
        ppl0 = eval_ppl_val(g_val, g_val_len, OPT_EVAL);
        printf("PPL inicial: %.2f\n", ppl0);
    }

    clock_t t0 = clock();
    /* entrena en chunks de 100 steps con cosine+clip (schedule sobre total) */
    int hechos = 0;
    while (hechos < OPT_STEPS) {
        int n = (OPT_STEPS - hechos > 100) ? 100 : OPT_STEPS - hechos;
        double loss = train_chunk(n, OPT_BATCH, OPT_LR, OPT_STEPS);
        hechos += n;
        printf("step %d/%d  loss=%.4f\n", hechos, OPT_STEPS, (float)loss);
        fflush(stdout);
    }
    double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;

#ifdef USE_GPU
    /* download weights from GPU before eval/save */
    if (OPT_GPU) {
        ad_gpu_download(&M);
        ad_gpu_free();
        printf("GPU: pesos descargados, device liberado\n");
    }
#endif

    /* PPL final + gate de publicacion */
    if (g_val) ppl1 = eval_ppl_val(g_val, g_val_len, OPT_EVAL);
    printf("PPL final: %.2f (antes %.2f)\n", ppl1, ppl0);
    int publish = (ppl0 == 0.0) || (ppl1 < ppl0);
    if (publish) {
        M.cfg.train_steps = M.cfg.train_steps + (uint32_t)OPT_STEPS;
        if (!ad_save(&M, out_path)) {
            printf("PUBLICADO: %s (mejora %.2f -> %.2f)\n", out_path, ppl0, ppl1);
        } else {
            printf("error al guardar %s\n", out_path);
            publish = 0;
        }
    } else {
        printf("RECHAZADO: ppl no mejoro, no se publica\n");
    }
    /* metrics.csv
     * columnas: fecha,modelo,steps,ppl0,ppl1,published,params,bytes,dim,L
     * (params/bytes permiten graficar la evolucion del tamano) */
    {
        FILE *f = fopen("metrics.csv", "a");
        if (f) {
            time_t now = time(NULL);
            char ds[32];
            strftime(ds, sizeof ds, "%Y-%m-%d %H:%M:%S", localtime(&now));
            size_t nparams = ad_total_floats(&M.cfg);
            size_t nbytes = AD_HDR + g_nfloats * sizeof(float);
            fprintf(f, "%s,%s,%d,%.4f,%.4f,%d,%zu,%zu,%d\n", ds, out_path,
                    OPT_STEPS, ppl0, ppl1, publish, nparams, nbytes,
                    M.cfg.dim);
            fclose(f);
        }
    }
    printf("listo (%.1fs)\n", secs);
    free(GRAD); free(ADM); free(ADV);
    ad_model_free(&M);
    return 0;
}

/* ==================================================================
 *  win_fwd_bwd: forward completo de la ventana + (backward)
 *  layout del archivo: tok_emb, pos_emb, por capa {ln1g ln1b Wqkv bqkv
 *  Wproj bproj ln2g ln2b W1 b1 W2 b2}, tail {lnfg lnfb Whead bhead}
 * ================================================================== */
static float win_fwd_bwd(AdModel *m, const int *bts, int T, int do_bwd) {
#ifdef USE_GPU
    if (OPT_GPU) {
        return ad_gpu_fwd_bwd(m, bts, T, do_bwd);
    }
#endif
    const int dim = m->cfg.dim, hd = m->cfg.head_dim, nh = m->cfg.n_heads;
    const int hid = m->cfg.hidden;
    const size_t d = (size_t)dim;
    float loss = 0.f;
    const float *WE = m->w;

    /* x[0][t] = tok_emb + pos_emb */
    for (int t = 0; t < T; t++) {
        int id = bts[t] % (m->cfg.vocab > 0 ? m->cfg.vocab : 256);
        const float *te = WE + (size_t)id * bD;
        float *x0 = XROW(0, t);
        for (int i = 0; i < dim; i++)
            x0[i] = te[i] + m->w[m->lay.pos_emb + (size_t)t * d + i];
    }

    /* bucle de capas */
    for (int l = 0; l < m->cfg.n_layers; l++) {
        size_t lb = m->lay.layers + (size_t)l * m->lay.per_layer;
        const float *ln1g = m->w + lb;
        const float *ln1b = ln1g + d;
        const float *Wqkv = ln1b + d;
        const float *bqkv = Wqkv + d * 3 * d;
        const float *Wp   = bqkv + 3 * d;
        const float *bp   = Wp   + d * d;
        const float *ln2g = bp + d;
        const float *ln2b = ln2g + d;
        const float *W1   = ln2b + d;
        const float *b1   = W1 + d * (size_t)hid;
        const float *W2   = b1 + hid;
        const float *b2   = W2 + (size_t)hid * d;

        float *x_in  = bX + (size_t)l * bT * bD;
        float *x_out = bX + (size_t)(l + 1) * bT * bD;
        float *l1o   = bL1 + (size_t)l * bT * bD;
        float *qkv   = bQKV + (size_t)l * bT * 3 * bD;
        float *ao    = bAO + (size_t)l * bT * bD;
        float *pj    = bPR + (size_t)l * bT * bD;
        float *l2o   = bL2 + (size_t)l * bT * bD;
        float *f1b_  = bF1 + (size_t)l * bT * bH;

        for (int t = 0; t < T; t++) {
            /* LN1 */
            float m1, s1;
            ln_fwd(l1o + (size_t)t * bD, x_in + (size_t)t * bD,
                   ln1g, ln1b, dim, &m1, &s1);
            /* QKV */
            ad_matmul(qkv + (size_t)t * 3 * bD, Wqkv,
                      l1o + (size_t)t * bD, 3 * dim, dim);
            for (int i = 0; i < 3 * dim; i++)
                qkv[(size_t)t * 3 * bD + i] += bqkv[i];

            /* atencion causal por cabezas: ao[t] = concat(softmax(q.s))
               k,v de s<=t leidos de qkv[s] (mismo bloque ya calculado) */
            const float *q = qkv + (size_t)t * 3 * bD;
            float *ao_t = ao + (size_t)t * bD;
            memset(ao_t, 0, bD * sizeof(float));
            for (int h = 0; h < nh; h++) {
                const float *qh = q + (size_t)h * hd;
                float *sc = bSCR;
                for (int s = 0; s <= t; s++) {
                    const float *kh = qkv + (size_t)s * 3 * bD + (size_t)d + (size_t)h * hd;
                    float dp2 = 0.f;
                    for (int i = 0; i < hd; i++) dp2 += qh[i] * kh[i];
                    sc[s] = dp2 * (1.0f / sqrtf((float)hd));
                }
                ad_softmax(sc, t + 1);
                for (int s = 0; s <= t; s++) {
                    const float *vh = qkv + (size_t)s * 3 * bD + 2 * bD + (size_t)h * hd;
                    float wgt = sc[s];
                    for (int i = 0; i < hd; i++) ao_t[(size_t)h * hd + i] += wgt * vh[i];
                }
            }
            /* proj + residual: x_out[t] = x_in[t] + Wproj*ao[t] + bp */
            ad_matmul(pj + (size_t)t * bD, Wp, ao_t, dim, dim);
            for (int i = 0; i < dim; i++)
                x_out[(size_t)t * bD + i] = x_in[(size_t)t * bD + i]
                                          + pj[(size_t)t * bD + i] + bp[i];
            /* LN2 + FFN + residual */
            float m2, s2;
            ln_fwd(l2o + (size_t)t * bD, x_out + (size_t)t * bD,
                   ln2g, ln2b, dim, &m2, &s2);
            ad_matmul(f1b_ + (size_t)t * bH, W1, l2o + (size_t)t * bD, hid, dim);
            for (int i = 0; i < hid; i++)
                f1b_[(size_t)t * bH + i] += b1[i];
            ad_gelu(f1b_ + (size_t)t * bH, hid);
            ad_matmul(pj + (size_t)t * bD, W2, f1b_ + (size_t)t * bH, dim, hid);
            for (int i = 0; i < dim; i++)
                x_out[(size_t)t * bD + i] += pj[(size_t)t * bD + i] + b2[i];
        }
    }

    /* LN final + head -> logits[t] predice bytes[t+1]
     * FASE D parcial: el head es la GEMM mas grande [T x dim]@[dim x V]
     * (V=2048 en BPE): una sola GEMM en GPU cuando esta disponible */
    const float *lfg = m->w + m->lay.lnf_g;
    const float *lfb = m->w + m->lay.lnf_b;
    const float *Wh  = m->w + m->lay.w_head;
    const float *bh  = m->w + m->lay.b_head;
    float *x_last = bX + (size_t)m->cfg.n_layers * bT * bD;
    for (int t = 0; t < T; t++) {
        float ml, sl;
        ln_fwd(bLF + (size_t)t * bD, x_last + (size_t)t * bD, lfg, lfb, dim, &ml, &sl);
    }
    if (ad_gpub_ok()) {
        /* GEMM batch en GPU: [T x dim] @ Whead^T + bh  (Whead es [V x dim]) */
        ad_gpub_lin(bLG, bLF, Wh, bh, T, dim, AD_VOCAB);
    } else {
        for (int t = 0; t < T; t++) {
            ad_matmul(bLG + (size_t)t * AD_VOCAB, Wh,
                      bLF + (size_t)t * bD, AD_VOCAB, dim);
            for (int i = 0; i < AD_VOCAB; i++)
                bLG[(size_t)t * AD_VOCAB + i] += bh[i];
        }
    }

    /* ---- loss: CE de logits[t] -> byte target = bytes[t+1] ---- */
    if (T < 2) return 0.f;
    double ce = 0.0;
    for (int t = 0; t < T - 1; t++) {
        const float *lg = bLG + (size_t)t * AD_VOCAB;
        float mx = lg[0];
        for (int i = 1; i < AD_VOCAB; i++) if (lg[i] > mx) mx = lg[i];
        double s = 0.0;
        for (int i = 0; i < AD_VOCAB; i++) s += exp((double)lg[i] - mx);
        int tgt = bts[t + 1] % (m->cfg.vocab > 0 ? m->cfg.vocab : 256);
        ce += -((double)lg[tgt] - mx - log(s));
    }
    loss = (float)(ce / (T - 1));
    if (!do_bwd) return loss;

    /* ==================== BACKWARD ==================== */
    /* dlogits = softmax(logits) - onehot(target), por posicion t<T-1 */
    float inv_b = 1.0f / (float)(T - 1);
    for (int t = 0; t < T - 1; t++) {
        float mx = bLG[(size_t)t * AD_VOCAB];
        for (int i = 1; i < AD_VOCAB; i++) {
            float v = bLG[(size_t)t * AD_VOCAB + i];
            if (v > mx) mx = v;
        }
        double s = 0.0;
        for (int i = 0; i < AD_VOCAB; i++)
            s += exp((double)bLG[(size_t)t * AD_VOCAB + i] - mx);
        float inv_s = (float)(1.0 / s);
        int tgt = bts[t + 1] % (m->cfg.vocab > 0 ? m->cfg.vocab : 256);
        float *dlt = bDLG + (size_t)t * AD_VOCAB;
        for (int i = 0; i < AD_VOCAB; i++) {
            float pi = expf(bLG[(size_t)t * AD_VOCAB + i] - mx) * inv_s;
            dlt[i] = (pi - (i == tgt ? 1.f : 0.f)) * inv_b;
        }
    }

    /* head: dWh[v,i] += dlog[v]*h[v];  dh = Wh^T . dlog;  dbh += dlog */
    {
        const float *Wh = m->w + m->lay.w_head;
        float *dWh = GRAD + m->lay.w_head;
        float *dbh = GRAD + m->lay.b_head;
        memset(bDX, 0, (size_t)bT * bD * sizeof(float));
        for (int t = 0; t < T - 1; t++) {
            const float *dl = bDLG + (size_t)t * AD_VOCAB;
            const float *h = bLF + (size_t)t * bD;
            gWadd(dWh, dl, h, AD_VOCAB, dim);
            for (int i = 0; i < AD_VOCAB; i++) dbh[i] += dl[i];
            /* dh = Wh^T * dl */
            ad_matmul_T(bDX + (size_t)t * bD, Wh, dl, AD_VOCAB, dim);
        }
    }
    return loss;
}





/* ==================================================================
 *  API WASM: entrenamiento EN VIVO desde el navegador
 *  MISMAS funciones que el CLI (train_chunk/eval_ppl_val) ? no un
 *  duplicado degradado. El corpus llega por buffer, los pesos mejorados
 *  se exportan a un buffer que JS puede serializar y descargar.
 *
 *   ad_live_start(model_buf, model_len, corpus, corpus_len, ctx)
 *   ad_live_chunk(steps, batch, lr) -> loss media
 *   ad_live_eval()                  -> ppl sobre el corpus en memoria
 *   ad_live_save(buf)               -> vuelca el modelo padre (ADMF completo)
 *   ad_live_steps()                 -> pasos acumulados de Adam
 * ================================================================== */
#ifdef __EMSCRIPTEN__
#include <emscripten.h>

static int live_loaded = 0;

EMSCRIPTEN_KEEPALIVE
int ad_live_start(const uint8_t *model_buf, int model_len,
                  const uint8_t *corpus, int corpus_len, int ctx) {
    if (ad_load_from(&M, model_buf, model_len) != 0) return -1;
    if (corpus_len < ctx + 2) return -2;
    g_train = (uint8_t *)malloc((size_t)corpus_len);
    if (!g_train) return -3;
    memcpy((void *)g_train, corpus, (size_t)corpus_len);
    g_train_len = (size_t)corpus_len;
    g_val = g_train;                 /* eval sobre el mismo corpus (didactico) */
    g_val_len = g_train_len;
    srand(12345);                    /* reproducible en el navegador */
    if (train_setup()) return -4;
    live_loaded = 1;
    return 0;
}

/* variante WASM: reusa el modelo YA cargado en g_m (adaptive.c) y copia
 * sus pesos a M, evitando re-descargar el .adm */
extern void *ad_global_model(void);   /* implementado abajo via helper */

EMSCRIPTEN_KEEPALIVE
int ad_live_start_from_loaded(int corpus_len, const uint8_t *corpus, int ctx) {
    /* g_m es el modelo residente que carg?? ad_load_mem: lo localizamos
     * re-exportando sus bytes via ad_live_save-like. Simplificacion:
     * train.c NO toca g_m; el worker pasa el buffer del modelo original.
     * => esta funcion NO se usa; el worker llama ad_live_start con el
     * ArrayBuffer descargado que ya tiene guardado. */
    (void)corpus_len; (void)corpus; (void)ctx;
    return -9;
}

EMSCRIPTEN_KEEPALIVE
double ad_live_chunk(int steps, int batch, float lr) {
    if (!live_loaded) return -1.0;
    return train_chunk(steps, batch, lr, 0); /* live: sin schedule */
}

EMSCRIPTEN_KEEPALIVE
double ad_live_eval(void) {
    if (!live_loaded || !g_val) return 0.0;
    return eval_ppl_val(g_val, g_val_len, 30);
}

EMSCRIPTEN_KEEPALIVE
int ad_live_steps(void) { return g_adam_t; }

/* exporta: cabecera ADM + pesos (mismo formato que ad_save, via buffer) */
EMSCRIPTEN_KEEPALIVE
int ad_live_save(uint8_t *out, int out_max) {
    if (!live_loaded) return -1;
    size_t need = AD_HDR + g_nfloats * sizeof(float);
    if ((size_t)out_max < need) return -2;
    uint32_t hdr[9] = {
        0x464D4441u, 1u,
        (uint32_t)M.cfg.dim, (uint32_t)M.cfg.n_layers,
        (uint32_t)M.cfg.n_heads, (uint32_t)M.cfg.hidden,
        (uint32_t)M.cfg.max_seq, M.cfg.train_steps, M.cfg.flags
    };
    memcpy(out, hdr, AD_HDR);   /* 36 bytes = 9 * 4 */
    memcpy(out + AD_HDR, M.w, g_nfloats * sizeof(float));
    M.cfg.train_steps += (uint32_t)g_adam_t;
    return (int)need;
}
#endif






