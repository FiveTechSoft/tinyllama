/* cma.c - Checkpoint Model Averaging (PuRo-2B 3.4)
 * promedia N .adm con pesos iguales -> out.adm
 *   cma.exe out.adm ck1.adm ck2.adm [ck3...]
 */
#include "adaptive.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "uso: cma out.adm ck1.adm ck2.adm [ck3...]\n");
        return 1;
    }
    AdModel acc;
    memset(&acc, 0, sizeof acc);
    int nacc = 0;
    for (int a = 2; a < argc; a++) {
        AdModel t;
        memset(&t, 0, sizeof t);
        int r = ad_load(&t, argv[a]);
        if (r) { fprintf(stderr, "skip %s (r=%d)\n", argv[a], r); continue; }
        if (!acc.w) {
            acc = t;      /* primer modelo adoptado (t ya no se libera) */
            continue;
        }
        if (t.lay.total != acc.lay.total) { ad_model_free(&t); continue; }
        for (size_t i = 0; i < acc.lay.total; i++) acc.w[i] += t.w[i];
        ad_model_free(&t);
        nacc++;
    }
    if (!acc.w || nacc == 0) { fprintf(stderr, "sin modelos validos\n"); return 1; }
    /* el primero no paso por el += ... */
    for (size_t i = 0; i < acc.lay.total; i++) acc.w[i] /= (float)(nacc + 1);
    acc.cfg.train_steps += (uint32_t)nacc;   /* metadato */
    if (ad_save(&acc, argv[2])) { fprintf(stderr, "save fail\n"); return 1; }
    printf("CMA: %d checkpoints promediados -> %s\n", nacc + 1, argv[2]);
    return 0;
}