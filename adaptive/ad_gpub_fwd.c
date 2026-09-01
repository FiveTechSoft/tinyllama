/* ad_gpub_fwd.c - forward batch en GPU (FASE B)
 *
 * reemplaza el forward de eval_ppl_val por la version batch:
 * los 4 matmuls por capa (QKV, proj, FFN1, FFN2) son UNA GEMM c/u sobre
 * [T x dim]; la atencion causal queda en CPU (depende de t).
 * verificacion: comparar logits contra el forward CPU para el mismo input.
 *
 *   ad_gpub_forward(m, bytes, T, logits_out)  -> 0 ok
 */
#include "adaptive.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int ad_gpub_init(void);
int ad_gpub_lin(float *out, const float *in, const float *W, const float *bias,
                int BT, int K, int N);

/* forward completo batch de una ventana de T bytes -> logits [T x V] */
int ad_gpub_forward(AdModel *m, const unsigned char *bytes, int T,
                    float *logits_out) {
    int dim = m->cfg.dim, hd = m->cfg.head_dim, nh = m->cfg.n_heads;
    int hid = m->cfg.hidden;
    size_t d = (size_t)dim, per = m->lay.per_layer;
    size_t V = AD_VOCAB;

    /* 1. embeddings: x[t] = tok_emb[id] + pos_emb[t] (CPU, memcpy) */
    float *X = (float *)malloc((size_t)T * d * sizeof(float));
    for (int t = 0; t < T; t++) {
        int id = bytes[t] & 0xFF;
        for (int i = 0; i < dim; i++)
            X[(size_t)t * d + i] = m->w[m->lay.tok_emb + (size_t)id * d + i]
                                 + m->w[m->lay.pos_emb + (size_t)t * d + i];
    }

    float *L1 = (float *)malloc((size_t)T * d * sizeof(float));
    float *QKV = (float *)malloc((size_t)T * 3 * d * sizeof(float));
    float *AO = (float *)malloc((size_t)T * d * sizeof(float));
    float *PR = (float *)malloc((size_t)T * d * sizeof(float));
    float *L2 = (float *)malloc((size_t)T * d * sizeof(float));
    float *F1 = (float *)malloc((size_t)T * (size_t)hid * sizeof(float));
    float *SCR = (float *)malloc((size_t)T * sizeof(float));

    for (int l = 0; l < m->cfg.n_layers; l++) {
        size_t lb = m->lay.layers + (size_t)l * per;
        const float *ln1g = m->w + lb, *ln1b = ln1g + d;
        const float *Wqkv = ln1b + d, *bqkv = Wqkv + d * 3 * d;
        const float *Wp = bqkv + 3 * d, *bp = Wp + d * d;
        const float *ln2g = bp + d, *ln2b = ln2g + d;
        const float *W1 = ln2b + d, *b1 = W1 + d * (size_t)hid;
        const float *W2 = b1 + hid, *b2 = W2 + (size_t)hid * d;

        /* LN1 batch en CPU */
        for (int t = 0; t < T; t++)
            ad_layernorm(L1 + (size_t)t * d, X + (size_t)t * d, ln1g, ln1b, dim);

        /* QKV batch en GPU: [T x dim] @ Wqkv[dim x 3dim] */
        ad_gpub_lin(QKV, L1, Wqkv, bqkv, T, dim, 3 * dim);

        /* atencion causal en CPU */
        for (int t = 0; t < T; t++) {
            const float *q = QKV + (size_t)t * 3 * d;
            float *ao_t = AO + (size_t)t * d;
            memset(ao_t, 0, d * sizeof(float));
            for (int h = 0; h < nh; h++) {
                const float *qh = q + (size_t)h * hd;
                for (int s = 0; s <= t; s++) {
                    const float *kh = QKV + (size_t)s * 3 * d + d + (size_t)h * hd;
                    float dp = 0.f;
                    for (int i = 0; i < hd; i++) dp += qh[i] * kh[i];
                    SCR[s] = dp / sqrtf((float)hd);
                }
                ad_softmax(SCR, t + 1);
                for (int s = 0; s <= t; s++) {
                    const float *vh = QKV + (size_t)s * 3 * d + 2 * d + (size_t)h * hd;
                    float wgt = SCR[s];
                    for (int i = 0; i < hd; i++) ao_t[(size_t)h * hd + i] += wgt * vh[i];
                }
            }
        }

        /* proj batch en GPU + residual */
        ad_gpub_lin(PR, AO, Wp, bp, T, dim, dim);
        for (int t = 0; t < T; t++)
            for (int i = 0; i < dim; i++)
                X[(size_t)t * d + i] += PR[(size_t)t * d + i];

        /* LN2 batch */
        for (int t = 0; t < T; t++)
            ad_layernorm(L2 + (size_t)t * d, X + (size_t)t * d, ln2g, ln2b, dim);

        /* FFN batch: W1 + gelu en CPU + W2 en GPU */
        ad_gpub_lin(F1, L2, W1, b1, T, dim, hid);
        ad_gelu(F1, T * hid);
        ad_gpub_lin(PR, F1, W2, b2, T, hid, dim);
        for (int t = 0; t < T; t++)
            for (int i = 0; i < dim; i++)
                X[(size_t)t * d + i] += PR[(size_t)t * d + i];
    }

    /* LN final + head batch en GPU */
    for (int t = 0; t < T; t++)
        ad_layernorm(L1 + (size_t)t * d, X + (size_t)t * d,
                     m->w + m->lay.lnf_g, m->w + m->lay.lnf_b, dim);
    ad_gpub_lin(logits_out, L1, m->w + m->lay.w_head, m->w + m->lay.b_head,
                T, dim, (int)V);

    free(X); free(L1); free(QKV); free(AO); free(PR); free(L2); free(F1); free(SCR);
    return 0;
}