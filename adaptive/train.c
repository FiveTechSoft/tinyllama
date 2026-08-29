/* train.c - entrenador del LLM adaptativo: backprop + Adam + gate PPL
 * (se construye por bloques: este es el bloque 1/5: includes + CLI)
 */
#include "ad_train.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ---- opciones globales ---- */
static int    OPT_STEPS = 2000;
static int    OPT_T     = 96;
static float  OPT_LR    = 3e-4f;
static int    OPT_EVAL  = 100;
static int    OPT_BATCH = 8;

static uint8_t *read_file(const char *path, size_t *len);

/* ==================================================================
 *  FORWARD + BACKWARD de una ventana de T bytes
 *  targets = bytes[t+1]; usa buffers bX/bL1/... ya allocados
 *  acumula gradientes en GRAD (arena espejo de M.w)
 *  devuelve CE media por prediccion (T-1)
 * ================================================================== */
static float win_fwd_bwd(AdModel *m, const uint8_t *bytes, int T, int do_bwd);

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

static AdModel M;               /* modelo en entrenamiento */
static float  *GRAD, *ADM, *ADV;/* gradientes + moments Adam */

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
        }
    }
    if (!data_path || (!fresh_d && !model_in)) {
        fprintf(stderr,
            "uso: train --data corpus.bin (--fresh dim,H,L | --model m.adm)"
            " [--out model.adm] [--steps N] [--batch B] [--ctx T] [--lr F]"
            " --val val.bin\n");
        return 1;
    }

    /* ---- cargar o crear modelo ---- */
    srand((unsigned)time(NULL));
    if (fresh_d > 0) {
        if (ad_init_fresh(&M, fresh_d, fresh_l, fresh_h, OPT_T * 2)) {
            fprintf(stderr, "no se pudo crear modelo fresh\n");
            return 2;
        }
        printf("modelo fresco: dim=%d L=%d H=%d\n", M.cfg.dim, fresh_l, fresh_h);
    } else {
        int r = ad_load(&M, model_in);
        if (r) { fprintf(stderr, "load %s -> %d\n", model_in, r); return 2; }
        printf("modelo cargado: steps=%u\n", M.cfg.train_steps);
    }

    /* ---- corpus ---- */
    size_t train_len = 0, val_len = 0;
    uint8_t *train = read_file(data_path, &train_len);
    uint8_t *val = val_path ? read_file(val_path, &val_len) : NULL;
    if (!train || train_len < (size_t)OPT_T + 2) {
        fprintf(stderr, "corpus %s vacio o pequeno\n", data_path);
        return 3;
    }
    if (val && val_len < (size_t)OPT_T + 2) { free(val); val = NULL; }

    /* ---- buffers + gradientes + Adam ---- */
    bT = OPT_T; bD = M.cfg.dim; bH = M.cfg.hidden;
    bHD = M.cfg.head_dim; bNH = M.cfg.n_heads; bL = M.cfg.n_layers;
    if (bufs_alloc()) { fprintf(stderr, "OOM buffers\n"); return 3; }
    GRAD = (float *)calloc(ad_total_floats(&M.cfg), sizeof(float));
    ADM  = (float *)calloc(ad_total_floats(&M.cfg), sizeof(float));
    ADV  = (float *)calloc(ad_total_floats(&M.cfg), sizeof(float));
    if (!GRAD || !ADM || !ADV) { fprintf(stderr, "OOM adam\n"); return 3; }

    printf("b2: modelo+corpus+buffers OK (train=%zu val=%zu)\n",
           train_len, val ? val_len : 0);

    /* ================== BUCLE DE ENTRENAMIENTO ================== */
    size_t nfloats = ad_total_floats(&M.cfg);
    double ppl0 = 0.0, ppl1 = 0.0;

    /* PPL inicial (eval) */
    if (val) {
        double tot = 0.0;
        for (int w = 0; w < OPT_EVAL; w++) {
            size_t st = (size_t)((double)ad_randf()
                       * (double)(val_len - (size_t)OPT_T - 1));
            tot += win_fwd_bwd(&M, val + st, OPT_T, 0);
        }
        ppl0 = exp((double)(tot / OPT_EVAL));
        printf("PPL inicial: %.2f\n", ppl0);
    }

    clock_t t0 = clock();
    for (int step = 1; step <= OPT_STEPS; step++) {
        double avg = 0.0;
        for (int bi = 0; bi < OPT_BATCH; bi++) {
            size_t st = (size_t)(ad_randf()
                        * (double)(train_len - (size_t)OPT_T - 1));
            avg += win_fwd_bwd(&M, train + st, OPT_T, 1);
        }
        /* Adam: promedia el gradiente por B y actualiza w */
        static int adam_t = 0;
        adam_t++;
        float bc1 = 1.0f - powf(0.9f, (float)adam_t);
        float bc2 = 1.0f - powf(0.999f, (float)adam_t);
        for (size_t i = 0; i < nfloats; i++) {
            float gi = GRAD[i] / (float)OPT_BATCH;
            ADM[i] = 0.9f * ADM[i] + 0.1f * gi;
            ADV[i] = 0.999f * ADV[i] + 0.001f * gi * gi;
            M.w[i] -= OPT_LR * (ADM[i] / bc1)
                    / (sqrtf(ADV[i] / bc2) + 1e-8f);
            GRAD[i] = 0.f;
        }
        if (step % 100 == 0 || step == OPT_STEPS) {
            printf("step %d/%d  loss=%.4f\n", step, OPT_STEPS, (float)(avg / OPT_BATCH));
            fflush(stdout);
        }
    }
    double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;

    /* PPL final + gate de publicacion */
    if (val) {
        double tot = 0.0;
        for (int w = 0; w < OPT_EVAL; w++) {
            size_t st = (size_t)(ad_randf()
                        * (double)(val_len - (size_t)OPT_T - 1));
            tot += win_fwd_bwd(&M, val + st, OPT_T, 0);
        }
        ppl1 = exp(tot / OPT_EVAL);
    }
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
    /* metrics.csv */
    {
        FILE *f = fopen("metrics.csv", "a");
        if (f) {
            time_t now = time(NULL);
            char ds[32];
            strftime(ds, sizeof ds, "%Y-%m-%d %H:%M:%S", localtime(&now));
            fprintf(f, "%s,%s,%d,%.4f,%.4f,%d\n", ds, out_path, OPT_STEPS,
                    ppl0, ppl1, publish);
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
static float win_fwd_bwd(AdModel *m, const uint8_t *bts, int T, int do_bwd) {
    const int dim = m->cfg.dim, hd = m->cfg.head_dim, nh = m->cfg.n_heads;
    const int hid = m->cfg.hidden;
    const size_t d = (size_t)dim;
    float loss = 0.f;
    const float *WE = m->w;

    /* x[0][t] = tok_emb + pos_emb */
    for (int t = 0; t < T; t++) {
        int id = bts[t] & 0xFF;
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

    /* LN final + head -> logits[t] predice bytes[t+1] */
    const float *lfg = m->w + m->lay.lnf_g;
    const float *lfb = m->w + m->lay.lnf_b;
    const float *Wh  = m->w + m->lay.w_head;
    const float *bh  = m->w + m->lay.b_head;
    float *x_last = bX + (size_t)m->cfg.n_layers * bT * bD;
    for (int t = 0; t < T; t++) {
        float ml, sl;
        ln_fwd(bLF + (size_t)t * bD, x_last + (size_t)t * bD, lfg, lfb, dim, &ml, &sl);
        ad_matmul(bLG + (size_t)t * AD_VOCAB, Wh,
                  bLF + (size_t)t * bD, AD_VOCAB, dim);
        for (int i = 0; i < AD_VOCAB; i++)
            bLG[(size_t)t * AD_VOCAB + i] += bh[i];
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
        int tgt = bts[t + 1] & 0xFF;
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
        int tgt = bts[t + 1] & 0xFF;
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



