/*
 * ad_core.c - nucleo del modelo adaptativo
 * Kernels (matmul, layernorm, softmax, gelu + backward), layout de tensores,
 * alloc/free, carga y guardado del formato ADM.
 */
#include "adaptive.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ================= kernels ================= */

/* out[r] = W[r,:] . x   (W [R x C] row-major) */
/* ---- kernels SIMD (de FiveTechSoft/dreaming, llm_inference.c) ---- */
#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

static inline float ad_dot(const float *a, const float *b, int n) {
#if defined(__AVX2__)
    int i = 0;
    __m256 sum = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
    }
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 s4 = _mm_add_ps(lo, hi);
    __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    float r = _mm_cvtss_f32(s1);
    for (; i < n; i++) r += a[i] * b[i];
    return r;
#elif defined(__SSE2__)
    int i = 0;
    __m128 sum = _mm_setzero_ps();
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
    }
    float tmp[4];
    _mm_storeu_ps(tmp, sum);
    float r = tmp[0]+tmp[1]+tmp[2]+tmp[3];
    for (; i < n; i++) r += a[i] * b[i];
    return r;
#else
    float r = 0.f;
    for (int i = 0; i < n; i++) r += a[i] * b[i];
    return r;
#endif
}

void ad_matmul(float *out, const float *W, const float *x, int R, int C) {
    for (int r = 0; r < R; r++) {
        const float *wr = W + (size_t)r * C;
        float s = 0.f;
        for (int c = 0; c < C; c++) s += wr[c] * x[c];
        out[r] = s;
    }
}

/* out[c] = sum_r W[r,c] * x[r]   (gradiente dx de un linear) */
void ad_matmul_T(float *out, const float *W, const float *x, int R, int C) {
    for (int c = 0; c < C; c++) out[c] = 0.f;
    for (int r = 0; r < R; r++) {
        float xv = x[r];
        const float *wr = W + (size_t)r * C;
        for (int c = 0; c < C; c++) out[c] += wr[c] * xv;
    }
}

void ad_layernorm(float *out, const float *x, const float *g, const float *b, int n) {
    float m = 0.f;
    for (int i = 0; i < n; i++) m += x[i];
    m /= n;
    float v = 0.f;
    for (int i = 0; i < n; i++) { float d = x[i] - m; v += d * d; }
    v /= n;
    float s = 1.0f / sqrtf(v + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = (x[i] - m) * s * g[i] + b[i];
}

void ad_softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0.f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

void ad_gelu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

void ad_gelu_bwd(float *dx, const float *pre, int n) {
    const float k0 = 0.7978845608f, k1 = 0.044715f;
    for (int i = 0; i < n; i++) {
        float v = pre[i];
        float u = k0 * (v + k1 * v * v * v);
        float th = tanhf(u);
        float du = k0 * (1.0f + 3.0f * k1 * v * v);
        dx[i] *= 0.5f * (1.0f + th) + 0.5f * v * (1.0f - th * th) * du;
    }
}

float ad_randf(void) { return (float)((double)rand() / (double)RAND_MAX); }

float ad_gauss(void) {
    float u1;
    do { u1 = ad_randf(); } while (u1 < 1e-7f);
    float u2 = ad_randf();
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

/* ================= layout =================
 * orden de tensores en la arena:
 *   tok_emb[vocab*dim], pos_emb[max_seq*dim],
 *   por capa l (base = layers + l*per_layer):
 *     ln1g[d] ln1b[d] Wqkv[d*3d] bqkv[3d] Wproj[d*d] bproj[d]
 *     ln2g[d] ln2b[d] W1[d*hid] b1[hid] W2[hid*d] b2[d]
 * tail: lnfg[d] lnfb[d] Whead[vocab*dim] bhead[vocab]
 * vocab dinamico: c->vocab = 256 (byte-level) o 2048+ (BPE)
 */
void ad_layout_build(AdConfig *c, AdLayout *L) {
    size_t d = (size_t)c->dim, hid = (size_t)c->hidden;
    size_t V = (size_t)(c->vocab > 0 ? c->vocab : 256);
    L->tok_emb = 0;
    L->pos_emb = V * d;
    L->layers  = L->pos_emb + (size_t)c->max_seq * d;
    size_t rel = 2 * d                       /* ln1 g,b */
        + d * 3 * d + 3 * d                  /* Wqkv, bqkv */
        + d * d + d                          /* Wproj, bproj */
        + 2 * d                              /* ln2 g,b */
        + d * hid + hid                      /* W1, b1 */
        + hid * d + d;                       /* W2, b2 */
    L->per_layer = rel;
    L->lnf_g  = L->layers + (size_t)c->n_layers * rel;
    L->lnf_b  = L->lnf_g + d;
    L->w_head = L->lnf_b + d;
    L->b_head = L->w_head + V * d;
    L->total  = L->b_head + V;
}

size_t ad_total_floats(const AdConfig *c) {
    AdLayout L;
    ad_layout_build((AdConfig *)c, &L);
    return L.total;
}

/* ================= alloc / free / fresh ================= */

int ad_model_alloc(AdModel *m) {
    ad_layout_build(&m->cfg, &m->lay);
    size_t V = (size_t)(m->cfg.vocab > 0 ? m->cfg.vocab : 256);
    m->w      = (float *)calloc(m->lay.total, sizeof(float));
    m->logits = (float *)malloc((size_t)V * sizeof(float));
    m->tokens = (int *)malloc((size_t)m->cfg.max_seq * sizeof(int));
    if (!m->w || !m->logits || !m->tokens) return -1;
    m->n_tok = 0;
    m->n_prompt = 0;
    return 0;
}

void ad_model_free(AdModel *m) {
    if (!m) return;
    free(m->w); free(m->logits); free(m->tokens);
    memset(m, 0, sizeof(*m));
}

/* inicializacion estilo GPT-2: std 0.02 (escalado por capas), LN gamma=1 */
/* init GPT-2 (std 0.02/sqrt(2L)); gammas de LN a 1, biases a 0 (calloc)
 * modo BPE: bpe_vocab>0 fija cfg.vocab */
int ad_init_fresh(AdModel *m, int dim, int n_layers, int n_heads, int seq) {
    return ad_init_fresh_v(m, dim, n_layers, n_heads, seq, 256);
}

int ad_init_fresh_v(AdModel *m, int dim, int n_layers, int n_heads, int seq,
                    int vocab) {
    if (dim <= 0 || n_heads <= 0 || dim % n_heads) return -1;
    if (n_layers < 1 || n_layers > AD_MAX_LAYERS) return -2;
    if (seq < 2 || seq > AD_MAX_SEQ) return -3;
    if (vocab < 256 || vocab > AD_VOCAB_MAX) return -5;
    memset(m, 0, sizeof(*m));
    m->cfg.dim = dim;
    m->cfg.n_layers = n_layers;
    m->cfg.n_heads = n_heads;
    m->cfg.hidden = 4 * dim;
    m->cfg.max_seq = seq;
    m->cfg.head_dim = dim / n_heads;
    m->cfg.vocab = vocab;
    m->cfg.train_steps = 0;
    m->cfg.flags = 0;
    if (ad_model_alloc(m) != 0) return -4;

    float std = 0.02f / sqrtf(2.0f * (float)n_layers);
    size_t V = (size_t)vocab;
    /* tok_emb y Whead con gaussiana; todo lo demas queda a 0 (calloc):
     * biases 0, gammas LN 1 */
    for (size_t i = 0; i < V * (size_t)dim; i++)
        m->w[m->lay.tok_emb + i] = std * ad_gauss();
    for (size_t i = 0; i < (size_t)dim; i++)
        m->w[m->lay.lnf_g + i] = 1.0f;
    for (size_t i = 0; i < V * (size_t)dim; i++)
        m->w[m->lay.w_head + i] = std * ad_gauss();
    /* gammas por capa = 1: ln1g en offset 0 del bloque, ln2g tras
     * ln1g[d] ln1b[d] Wqkv[d*3d] bqkv[3d] Wproj[d*d] bproj[d] */
    size_t rel_ln2g = 2 * (size_t)dim
        + (size_t)dim * 3 * (size_t)dim + 3 * (size_t)dim
        + (size_t)dim * (size_t)dim + (size_t)dim;
    for (int l = 0; l < n_layers; l++) {
        size_t lb = m->lay.layers + (size_t)l * m->lay.per_layer;
        m->w[lb] = 1.0f;                 /* ln1g */
        m->w[lb + rel_ln2g] = 1.0f;      /* ln2g */
    }
    return 0;
}

/* ================= IO ================= */

static void put_u32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }

int ad_save(AdModel *m, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    put_u32(f, 0x464D4441u);              /* 'ADMF' */
    put_u32(f, AD_VERSION);
    put_u32(f, (uint32_t)m->cfg.dim);
    put_u32(f, (uint32_t)m->cfg.n_layers);
    put_u32(f, (uint32_t)m->cfg.n_heads);
    put_u32(f, (uint32_t)m->cfg.hidden);
    put_u32(f, (uint32_t)m->cfg.max_seq);
    put_u32(f, m->cfg.train_steps);
    put_u32(f, m->cfg.flags);
    fwrite(m->w, sizeof(float), m->lay.total, f);
    fclose(f);
    return 0;
}

int ad_load(AdModel *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -10;
    uint32_t v[9];
    if (fread(v, 4, 9, f) != 9) { fclose(f); return -11; }
    if (v[0] != 0x464D4441u) { fclose(f); return -12; }
    if (v[1] != AD_VERSION) { fclose(f); return -13; }
    AdConfig c;
    c.dim = (int)v[2]; c.n_layers = (int)v[3]; c.n_heads = (int)v[4];
    c.hidden = (int)v[5]; c.max_seq = (int)v[6];
    c.train_steps = v[7]; c.flags = 0;
    if (c.dim <= 0 || c.n_heads <= 0 || c.dim % c.n_heads) { fclose(f); return -16; }
    if (c.n_layers < 1 || c.n_layers > AD_MAX_LAYERS) { fclose(f); return -17; }
    if (c.max_seq < 2 || c.max_seq > AD_MAX_SEQ) { fclose(f); return -18; }
    c.head_dim = c.dim / c.n_heads;
    ad_model_free(m);
    memset(m, 0, sizeof(*m));
    m->cfg = c;
    if (ad_model_alloc(m) != 0) { fclose(f); return -19; }
    if (fread(m->w, sizeof(float), m->lay.total, f) != m->lay.total) {
        fclose(f);
        return -20;
    }
    fclose(f);
    return 0;
}

/* ================= inferencia ================= */

static float *s_x, *s_h, *s_qkv, *s_xb, *s_f1;
static float *s_kc, *s_vc;          /* [L][max_seq][dim] */
static size_t s_cache_bytes = 0;

static int infer_alloc(AdModel *m) {
    size_t d = (size_t)m->cfg.dim;
    size_t ms = (size_t)m->cfg.max_seq;
    size_t L = (size_t)m->cfg.n_layers;
    s_x   = (float *)malloc(d * sizeof(float));
    s_h   = (float *)malloc(d * sizeof(float));
    s_qkv = (float *)malloc(3 * d * sizeof(float));
    s_xb  = (float *)malloc(d * sizeof(float));
    s_f1  = (float *)malloc((size_t)m->cfg.hidden * sizeof(float));
    s_kc  = (float *)calloc(L * ms * d, sizeof(float));
    s_vc  = (float *)calloc(L * ms * d, sizeof(float));
    if (!s_x || !s_h || !s_qkv || !s_xb || !s_f1 || !s_kc || !s_vc) return -1;
    s_cache_bytes = L * ms * d * sizeof(float);
    return 0;
}

/* forward de UNA posicion (t = n_tok-1) usando KV cache; deja logits listos */
void ad_forward(AdModel *m) {
    int dim = m->cfg.dim, hd = m->cfg.head_dim, nh = m->cfg.n_heads;
    int hid = m->cfg.hidden;
    size_t d = (size_t)dim;
    int t = m->n_tok - 1;
    if (t < 0 || t >= m->cfg.max_seq) return;
    if (!s_kc && infer_alloc(m) != 0) return;

    const float *tok_emb = m->w + m->lay.tok_emb;
    float *x = s_x;
    int id = m->tokens[t] & 0xFF;
    for (int i = 0; i < dim; i++)
        x[i] = tok_emb[(size_t)id * d + i] + m->w[m->lay.pos_emb + (size_t)t * d + i];

    float *qkv = s_qkv;
    for (int l = 0; l < m->cfg.n_layers; l++) {
        size_t lb = m->lay.layers + (size_t)l * m->lay.per_layer;
        const float *ln1g  = m->w + lb;
        const float *ln1b  = ln1g + d;
        const float *Wqkv  = ln1b + d;
        const float *bqkv  = Wqkv + d * 3 * d;
        const float *Wproj = bqkv + 3 * d;
        const float *bproj = Wproj + d * d;
        const float *ln2g  = bproj + d;
        const float *ln2b  = ln2g + d;
        const float *W1    = ln2b + d;
        const float *b1    = W1 + d * (size_t)hid;
        const float *W2    = b1 + hid;
        const float *b2    = W2 + (size_t)hid * d;

        ad_layernorm(s_h, x, ln1g, ln1b, dim);
        ad_matmul(qkv, Wqkv, s_h, 3 * dim, dim);
        for (int i = 0; i < 3 * dim; i++) qkv[i] += bqkv[i];
        float *q = qkv, *k = qkv + dim, *v = qkv + 2 * dim;

        float *kc = s_kc + (size_t)l * (size_t)m->cfg.max_seq * d;
        float *vc = s_vc + (size_t)l * (size_t)m->cfg.max_seq * d;
        memcpy(kc + (size_t)t * d, k, d * sizeof(float));
        memcpy(vc + (size_t)t * d, v, d * sizeof(float));

        float scale = 1.0f / sqrtf((float)hd);
        memset(s_xb, 0, d * sizeof(float));
        for (int hh = 0; hh < nh; hh++) {
            const float *qh = q + (size_t)hh * hd;
            float *scr = s_f1;                       /* scratch [t+1] */
            for (int s = 0; s <= t; s++) {
                const float *ks = kc + (size_t)s * d + (size_t)hh * hd;
                float dp = 0.f;
                for (int i = 0; i < hd; i++) dp += qh[i] * ks[i];
                scr[s] = dp * scale;
            }
            ad_softmax(scr, t + 1);
            float *oh = s_xb + (size_t)hh * hd;
            for (int s = 0; s <= t; s++) {
                const float *vs = vc + (size_t)s * d + (size_t)hh * hd;
                float wgt = scr[s];
                for (int i = 0; i < hd; i++) oh[i] += wgt * vs[i];
            }
        }
        ad_matmul(s_qkv, Wproj, s_xb, dim, dim);
        for (int i = 0; i < dim; i++) x[i] += s_qkv[i] + bproj[i];

        /* FFN */
        ad_layernorm(s_h, x, ln2g, ln2b, dim);
        ad_matmul(s_f1, W1, s_h, m->cfg.hidden, dim);
        for (int i = 0; i < hid; i++) s_f1[i] += b1[i];
        ad_gelu(s_f1, hid);
        ad_matmul(s_xb, W2, s_f1, dim, m->cfg.hidden);
        for (int i = 0; i < dim; i++) x[i] += s_xb[i] + b2[i];
    }

    ad_layernorm(s_h, x, m->w + m->lay.lnf_g, m->w + m->lay.lnf_b, dim);
    {
        size_t V = (size_t)(m->cfg.vocab > 0 ? m->cfg.vocab : 256);
        ad_matmul(m->logits, m->w + m->lay.w_head, s_h, (int)V, dim);
        for (size_t i = 0; i < V; i++)
            m->logits[i] += m->w[m->lay.b_head + i];
    }
}

/* ---------------- API de conversacion ---------------- */

int ad_set_prompt(AdModel *m, const char *text) {
    if (!s_kc && infer_alloc(m) != 0) return -1;
    m->n_tok = 0;
    memset(s_kc, 0, s_cache_bytes);
    memset(s_vc, 0, s_cache_bytes);
    m->tokens[m->n_tok++] = AD_CTL_USER;
    for (const unsigned char *p = (const unsigned char *)text;
         *p && m->n_tok < m->cfg.max_seq - 2; p++)
        m->tokens[m->n_tok++] = *p;
    m->tokens[m->n_tok++] = AD_CTL_BOT;
    m->n_prompt = m->n_tok;
    for (int i = 1; i <= m->n_prompt; i++) {
        int save = m->n_tok;
        m->n_tok = i;
        ad_forward(m);
        m->n_tok = save;
    }
    return m->n_prompt;
}

int ad_step(AdModel *m, float temp, int top_k) {
    if (m->n_tok >= m->cfg.max_seq - 1) return -1;
    size_t V = (size_t)(m->cfg.vocab > 0 ? m->cfg.vocab : 256);
    static float *s_p = NULL; static int *s_idx = NULL;
    static size_t s_cap = 0;
    if (s_cap < V) {
        free(s_p); free(s_idx);
        s_p = (float *)malloc(V * sizeof(float));
        s_idx = (int *)malloc(V * sizeof(int));
        if (!s_p || !s_idx) return -2;
        s_cap = V;
    }
    float mx = m->logits[0];
    for (size_t i = 1; i < V; i++) if (m->logits[i] > mx) mx = m->logits[i];
    float sum = 0.f;
    for (size_t i = 0; i < V; i++) s_p[i] = expf((m->logits[i] - mx) / temp);
    for (size_t i = 0; i < V; i++) sum += s_p[i];
    for (size_t i = 0; i < V; i++) s_p[i] /= sum;

    for (size_t i = 0; i < V; i++) s_idx[i] = (int)i;
    for (size_t i = 1; i < V; i++) {                /* insertion sort desc */
        int idv = s_idx[i];
        size_t j = i - 1;
        while (j >= 0 && s_p[s_idx[j]] < s_p[idv]) { s_idx[j+1] = s_idx[j]; j--; }
        s_idx[j+1] = idv;
    }
    if (top_k <= 0 || top_k > (int)V) top_k = (int)V;
    float acc = 0.f;
    for (int i = 0; i < top_k; i++) acc += s_p[s_idx[i]];
    float r = ad_randf() * acc;
    int next = s_idx[top_k - 1];
    for (int i = 0; i < top_k; i++) {
        r -= s_p[s_idx[i]];
        if (r <= 0.f) { next = s_idx[i]; break; }
    }
    if (next == AD_CTL_EOS || next == AD_CTL_USER) return -1;
    m->tokens[m->n_tok++] = next;
    ad_forward(m);
    return next;
}

/* ================= API WASM (modelo en memoria) =================
 * JS descarga model.adm y pasa el buffer directamente. */
/* ================= API WASM (modelo en memoria) =================
 * JS descarga model.adm y pasa el buffer directamente. */

int ad_load_from(AdModel *m, const uint8_t *buf, int size) {
    if (!buf || size < AD_HDR || memcmp(buf, "ADMF", 4) != 0) return -1;
    const uint32_t *v = (const uint32_t *)buf;
    if (v[1] != AD_VERSION) return -2;
    ad_model_free(m);
    memset(m, 0, sizeof(*m));
    m->cfg.dim = (int)v[2];
    m->cfg.n_layers = (int)v[3];
    m->cfg.n_heads = (int)v[4];
    m->cfg.hidden = (int)v[5];
    m->cfg.max_seq = (int)v[6];
    m->cfg.train_steps = v[7];
    m->cfg.flags = v[8];
    if (m->cfg.dim <= 0 || m->cfg.n_heads <= 0 || m->cfg.dim % m->cfg.n_heads)
        return -3;
    if (m->cfg.n_layers < 1 || m->cfg.n_layers > AD_MAX_LAYERS) return -4;
    if (m->cfg.max_seq < 2 || m->cfg.max_seq > AD_MAX_SEQ) return -5;
    m->cfg.head_dim = m->cfg.dim / m->cfg.n_heads;
    if (ad_model_alloc(m) != 0) return -6;
    if ((size_t)size < AD_HDR + m->lay.total * sizeof(float)) return -7;
    memcpy(m->w, buf + AD_HDR, m->lay.total * sizeof(float));
    return 0;
}

#ifdef __EMSCRIPTEN__
#include <emscripten.h>

static AdModel g_m;

EMSCRIPTEN_KEEPALIVE
int ad_load_mem(const uint8_t *buf, int size) {
    return ad_load_from(&g_m, buf, size);
}

EMSCRIPTEN_KEEPALIVE
int ad_set_prompt_mem(const char *text) {
    return ad_set_prompt(&g_m, text);
}

EMSCRIPTEN_KEEPALIVE
int ad_step_mem(float temp, int top_k) {
    return ad_step(&g_m, temp, top_k);
}

EMSCRIPTEN_KEEPALIVE
int ad_n_prompt(void) { return g_m.n_prompt; }

EMSCRIPTEN_KEEPALIVE
int ad_n_tok(void) { return g_m.n_tok; }

EMSCRIPTEN_KEEPALIVE
int ad_dim(void) { return g_m.cfg.dim; }

EMSCRIPTEN_KEEPALIVE
static const char *ad_token_str_impl(AdModel *m, int id) {
    static char b[2];
    (void)m;
    b[0] = (char)(id & 0xFF);
    b[1] = 0;
    return b;
}

const char *ad_tok_str(int id) { return ad_token_str_impl(&g_m, id); }

/* ---- metricas de la version cargada ----
 * total_params   = lay.total (floats)
 * train_steps    = pasos de Adam del checkpoint
 * n_layers/heads/hidden/max_seq = config
 */
EMSCRIPTEN_KEEPALIVE
int ad_stats(int *out, int max_out) {
    if (!out || max_out < 8) return -1;
    out[0] = (int)g_m.lay.total;          /* pesos totales */
    out[1] = (int)g_m.cfg.dim;
    out[2] = (int)g_m.cfg.n_layers;
    out[3] = (int)g_m.cfg.n_heads;
    out[4] = (int)g_m.cfg.hidden;
    out[5] = (int)g_m.cfg.max_seq;
    out[6] = (int)g_m.cfg.train_steps;    /* pasos de entrenamiento */
    out[7] = 1;                            /* version de formato */
    return 8;
}

/* ---- gigakernel: genera N tokens en UNA llamada (sin dispatch JS/token) ----
 * out: buffer de N bytes JS->C; devuelve n de tokens escritos.
 * strings de control (USER/BOT/EOS) cortan la generacion. */
EMSCRIPTEN_KEEPALIVE
int ad_generate_n(const char *prompt, int max_tokens, float temp, int top_k,
                  uint8_t *out) {
    if (ad_set_prompt(&g_m, prompt) < 0) return -1;
    int n = 0;
    for (int i = 0; i < max_tokens; i++) {
        int id = ad_step(&g_m, temp, top_k);
        if (id < 0) break;
        if (out) out[n] = (uint8_t)id;
        n++;
    }
    return n;
}
#endif

