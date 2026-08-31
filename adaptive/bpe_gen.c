/* bpe_gen.c - genera texto del modelo BPE: prompt + N tokens con top5 debug
 *   bpe_gen.exe model.adm vocab.bpe "prompt" [n] [temp] [topk]
 */
#include "adaptive.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern int bpe_load(const char *path);
extern int bpe_encode(const char *text, int *ids, int max_ids);
extern int bpe_decode(const int *ids, int n, char *out, int outmax);

int main(int argc, char **argv) {
    const char *path = argv[1];
    const char *vp = argc >= 3 ? argv[2] : "vocab.bpe";
    const char *prompt = argc >= 4 ? argv[3] : "hola";
    int N = argc >= 5 ? atoi(argv[4]) : 60;
    float temp = argc >= 6 ? (float)atof(argv[5]) : 0.7f;
    int topk = argc >= 7 ? atoi(argv[6]) : 20;

    bpe_load(vp);
    srand(42);
    AdModel m; memset(&m, 0, sizeof m);
    if (ad_load(&m, path)) { fprintf(stderr, "load fail\n"); return 1; }
    printf("vocab=%d dim=%d\n", m.cfg.vocab, m.cfg.dim);

    /* tokeniza prompt manual (mismos controles que chat) */
    m.tokens[m.n_tok++] = AD_CTL_USER;
    int ids[256];
    int ni = bpe_encode(prompt, ids, 512);
    for (int i = 0; i < ni && m.n_tok < m.cfg.max_seq - 5; i++)
        m.tokens[m.n_tok++] = ids[i];
    m.tokens[m.n_tok++] = AD_CTL_BOT;

    /* prefill con debug de logits tras cada paso (solo los primeros 3) */
    for (int i = 1; i <= m.n_prompt; i++) {
        int save = m.n_tok;
        m.n_tok = i;
        ad_forward(&m);
        m.n_tok = save;
        if (i >= m.n_prompt - 2) {
            printf("pos %d: top5 logits: ", i);
            float *lg = m.logits;
            size_t V = m.cfg.vocab;
            for (int k = 0; k < 5; k++) {
                size_t be = 0; float bv = -1e30f;
                for (size_t j = 0; j < V; j++) {
                    int usado = 0;
                    for (int kk = 0; kk < k; kk++) {}
                    int skip = 0;
                    for (size_t z = 0; z < j; z++) { if (lg[j] < -9e29f) skip = 1; }
                    if (lg[j] > bv) { bv = lg[j]; be = j; }
                }
                lg[be] = -1e30f;
                printf("%ld:%.2f ", (long)be, bv);
            }
            printf("\n");
        }
    }
    /* genera */
    for (int i = 0; i < N; i++) {
        int id = ad_step(&m, temp, topk);
        if (id < 0) break;
        printf("%d ", id);
    }
    printf("\n");
    return 0;
}