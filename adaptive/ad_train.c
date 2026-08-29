/*
 * ad_train.c - forward + backward completos de una ventana, eval PPL
 */
#include "ad_train.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

WA      g_wa;
AdModel g_tm;

static void *xcalloc(size_t n, size_t s) {
    void *p = calloc(n, s);
    if (!p) { fprintf(stderr, "OOM\n"); exit(1); }
    return p;
}

int tr_alloc_work(AdModel *m) {
    int d = m->cfg.dim, T = m->cfg.max_seq;
    size_t L = (size_t)m->cfg.n_layers, hid = (size_t)m->cfg.hidden;
    if (T > 256) T = 256;
    size_t V = AD_VOCAB;
    g_wa.x       = (float *)xmalloc((L + 1) * T * d * sizeof(float));
    g_wa.ln1_out = (float *)xmalloc(L * T * d * sizeof(float));
    g_wa.qkv     = (float *)xmalloc(L * T * 3 * d * sizeof(float));
    g_wa.ao      = (float *)xmalloc(L * T * d * sizeof(float));
    g_wa.proj    = (float *)xmalloc(L * T * d * sizeof(float));
    g_wa.ln2_out = (float *)xmalloc(L * T * d * sizeof(float));
    g_wa.f1      = (float *)xmalloc(L * T * hid * sizeof(float));
    g_wa.f1a     = (float *)xmalloc(L * T * hid * sizeof(float));
    g_wa.lnf_out = (float *)xmalloc(T * d * sizeof(float));
    g_wa.logits  = (float *)xmalloc(T * V * sizeof(float));
    g_wa.dx      = (float *)xmalloc(T * d * sizeof(float));
    g_wa.dlogits = (float *)xmalloc(T * V * sizeof(float));
    g_wa.dh      = (float *)xmalloc(T * (hid > (size_t)d ? hid : (size_t)d) * sizeof(float));
    g_wa.scr     = (float *)xmalloc(T * sizeof(float));
    return 0;
}

void tr_free_work(void) {
    free(g_wa.x); free(g_wa.ln1_out); free(g_wa.qkv); free(g_wa.ao);
    free(g_wa.proj); free(g_wa.ln2_out); free(g_wa.f1); free(g_wa.f1a);
    free(g_wa.lnf_out); free(g_wa.logits);
    free(g_wa.dx); free(g_wa.dlogits); free(g_wa.dh); free(g_wa.scr);
    memset(&g_wa, 0, sizeof g_wa);
}

/* ---- ln backward (dx + acumuladores externos dg,db) ---- */
void tr_ln_bwd(const float *x, const float *dy, const float *g,
               float m, float s, int n, float *dx, float *dg, float *db);

float tr_window(AdModel *m, const uint8_t *bytes, int T, int do_bwd) {
    const int dim = m.cfg.dim, hd = (dim) / (m.cfg.n_heads), nh = m.cfg.n_heads;
    const int hid_v = m.cfg.hidden;
    const size_t d = (size_t)dim, hid = (size_t)hid_v, V = AD_VOCAB;
    const float *W = m->w;
    size_t LBASE = m->lay.layers, per = m->lay.per_layer;
    float inv_T = 1.0f / (float)T;

    /* embeddings */
    for (int t = 0; t < T; t++) {
        int id = bytes[t] & 0xFF;
        for (int i = 0; i < dim; i++)
            wa_x(m, 0, t)[i] = m->w[m->lay.tok_emb + (size_t)id * d + i]
                             + m->w[m->lay.pos_emb + (size_t)t * d + i];
    }
    (void)LBASE;
    return 0.f;
}