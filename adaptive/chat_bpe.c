/* chat_bpe.c - chat del modelo BPE (vocab>256): tokens -> vocab.bpe -> texto
 *
 *   chat_bpe.exe model.adm vocab.bpe [temp] [topk]
 */
#include "adaptive.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int bpe_load(const char *pathng);
int bpe_encode(const char *text, int *ids, int max_ids);
int bpe_decode(const int *ids, int n, char *out, int outmax);

int main(int argc, char **argv) {
    const char *path = argc >= 2 ? argv[1] : "model.adm";
    const char *vocab = argc >= 3 ? argv[2] : "vocab.bpe";
    float temp = argc >= 4 ? (float)atof(argv[3]) : 0.8f;
    int topk = argc >= 5 ? atoi(argv[4]) : 40;

    if (bpe_load(vocab) < 0) { fprintf(stderr, "no vocab %s\n", vocab); return 1; }

    srand((unsigned)time(NULL));
    fprintf(stderr, "[antes de load]\n"); fflush(stderr);
    AdModel m; memset(&m, 0, sizeof m);
    int r = ad_load(&m, path);
    fprintf(stderr, "[load=%d vocab=%d seq=%d]\n", r, m.cfg.vocab, m.cfg.max_seq);
    fflush(stderr);
    if (r) { fprintf(stderr, "no carga (%d)\n", r); return 1; }
    printf("chat BPE v1 | dim=%d L=%d vocab=%d | /salir\n",
           m.cfg.dim, m.cfg.n_layers, m.cfg.vocab);

    static char line[1024];
    printf("> ");
    while (fgets(line, sizeof line, stdin)) {
        size_t nl = strlen(line);
        if (nl && line[nl-1] == '\n') line[nl-1] = 0;
        if (!line[0]) { printf("> "); continue; }
        if (!strncmp(line, "/salir", 6)) break;
        /* encode BPE del user text + controles */
        fprintf(stderr, "[encodando '%s' vocab=%d]\n", line, m.cfg.vocab); fflush(stderr);
        m.tokens[m.n_tok++] = AD_CTL_USER;
        int ids[512];
        int nids = 0;
        if (m.cfg.vocab > 256) { nids = bpe_encode(line, ids, 512); }
        else { for (size_t i = 0; i < strlen(line); i++) ids[i] = (unsigned char)line[i]; nids = (int)strlen(line); }
        fprintf(stderr, "[encodeado: %d ids]\n", nids); fflush(stderr);
        for (int i = 0; i < nids && m.n_tok < m.cfg.max_seq - 10; i++)
            m.tokens[m.n_tok++] = ids[i];
        m.tokens[m.n_tok++] = AD_CTL_BOT;
        m.n_prompt = m.n_tok;
        /* el KV cache del core es estatico global: reusa infer_alloc con
           el cfg del modelo cargado; prefill */
        fprintf(stderr, "[prefill de %d tokens...]\n", m.n_tok); fflush(stderr);
        for (int i = 1; i <= m.n_prompt; i++) {
            int save = m.n_tok; m.n_tok = i; ad_forward(&m); m.n_tok = save;
        }
        fprintf(stderr, "[prefill OK]\n"); fflush(stderr);
        /* generacion + decode BPE */
        int gen[256] = {0};
        int gn = 0;
        for (int i = 0; i < 100; i++) {
            int id = ad_step(&m, temp, topk);
            fprintf(stderr, "[step %d -> %d]\n", i, id); fflush(stderr);
            if (id < 0) break;
            gen[gn++] = id;
            if (m.cfg.vocab <= 256) putchar((char)id);
        }
        if (m.cfg.vocab > 256) {
            char out[16384];
            bpe_decode(gen, gn, out, sizeof out);
            fputs(out, stdout);
        }
        printf("\n\n> ");
   }
    ad_model_free(&m);
    return 0;
}