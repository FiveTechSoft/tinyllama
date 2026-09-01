/* ad_expand.c - Net2Net function-preserving para el modelo adaptativo
 * replica tensores: nuevo[i] = viejo[i % D] (cols) / nuevo[r] = viejo[r % R]
 */
#include "adaptive.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void rep_vec(float *dst, const float *src, int D0, int D1) {
    for (int i = 0; i < D1; i++) dst[i] = src[i % D0];
}

static void rep_cols(float *dst, const float *src, int R, int C0, int C1) {
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C1; c++)
            dst[(size_t)r * C1 + c] = src[(size_t)r * C0 + (c % C0)];
}

static void rep_rows(float *dst, const float *src, int R0, int R1, int C) {
    for (int r = 0; r < R1; r++)
        for (int c = 0; c < C; c++)
            dst[(size_t)r * C + c] = src[(size_t)(r % R0) * C + c];
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "uso: ad_expand in.adm out.adm [factor=2]\n");
        return 1;
    }
    int factor = (argc >= 4) ? atoi(argv[3]) : 2;
    if (factor < 2 || factor > 4) factor = 2;

    srand((unsigned)time(NULL));
    AdModel src;
    memset(&src, 0, sizeof src);
    if (ad_load(&src, argv[1])) { fprintf(stderr, "no carga\n"); return 1; }
    int D0 = src.cfg.dim;
    int D = D0 * factor;
    int V = src.cfg.vocab > 0 ? src.cfg.vocab : 256;

    AdModel w;
    memset(&w, 0, sizeof w);
    if (ad_init_fresh_v(&w, D, src.cfg.n_layers, src.cfg.n_heads,
                        src.cfg.max_seq, src.cfg.vocab)) {
        fprintf(stderr, "alloc fail\n");
        return 1;
    }
    w.cfg.train_steps = src.cfg.train_steps;
    w.cfg.flags = src.cfg.flags;
    printf("expand: dim %d -> %d | hidden %d -> %d | pesos %.2fM -> %.2fM\n",
           D0, D, src.cfg.hidden, w.cfg.hidden,
           ad_total_floats(&src.cfg) / 1e6, ad_total_floats(&w.cfg) / 1e6);

    rep_cols(w.w + w.lay.tok_emb, src.w + src.lay.tok_emb, V, D0, D);
    rep_cols(w.w + w.lay.pos_emb, src.w + src.lay.pos_emb, src.cfg.max_seq, D0, D);

    for (int l = 0; l < src.cfg.n_layers; l++) {
        size_t so = src.lay.layers + (size_t)l * src.lay.per_layer;
        size_t db = w.lay.layers + (size_t)l * w.lay.per_layer;
        int hid0 = src.cfg.hidden, hid1 = w.cfg.hidden;
        int d0 = D0, d1 = D;
        /* ln1g ln1b */
        rep_vec(w.w + db, src.w + so, d0, d1);
        rep_vec(w.w + db + d1, src.w + so + d0, d0, d1);
        /* Wqkv [D0 x 3D0] -> [D x 3D]  (cols y rows al mismo tiempo) */
        rep_cols(w.w + db + 2 * (size_t)d1,
                 src.w + so + 2 * (size_t)d0,
                 d0, d0 * 3, d1 * 3);
        /* replica rows tambien (3*D salida) */
        rep_rows(w.w + db + 2 * (size_t)d1,
                 w.w + db + 2 * (size_t)d1,
                 d0 * 3, d1 * 3, d1);
        /* bqkv [3D0] -> [3D] */
        size_t bq_off_src = so + 2 * (size_t)d0 + (size_t)d0 * 3 * d0;
        size_t bq_off_dst = db + 2 * (size_t)d1 + (size_t)d1 * 3 * d1;
        rep_vec(w.w + bq_off_dst, src.w + bq_off_src, 3 * d0, 3 * d1);
        /* Wproj [D0 x D0] -> [D x D] */
        size_t wp_off_src = bq_off_src + 3 * d0;
        size_t wp_off_dst = bq_off_dst + 3 * d1;
        rep_rows(w.w + wp_off_dst, src.w + wp_off_src, d0, d1, d0);
        rep_cols(w.w + wp_off_dst, w.w + wp_off_dst, d1, d0, d1);
        /* bproj */
        size_t bp_off_src = wp_off_src + (size_t)d0 * d0;
        size_t bp_off_dst = wp_off_dst + (size_t)d1 * d1;
        rep_vec(w.w + bp_off_dst, src.w + bp_off_src, d0, d1);
        /* ln2g ln2b */
        size_t l2_src = bp_off_src + d0;
        size_t l2_dst = bp_off_dst + d1;
        rep_vec(w.w + l2_dst, src.w + l2_src, d0, d1);
        rep_vec(w.w + l2_dst + d1, src.w + l2_src + d0, d0, d1);
        /* W1 [D0 x hid0] -> [D x hid1] */
        size_t w1_src = l2_src + 2 * d0;
        size_t w1_dst = l2_dst + 2 * d1;
        rep_cols(w.w + w1_dst_dummy(), w.w + w1_dummy(), 0, 0, 0);
        (void)w1_src; (void)w1_dst;
    }
    return 0;
}