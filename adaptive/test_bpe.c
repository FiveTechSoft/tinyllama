/* test_bpe.c - verifica bpe.c: carga vocab, codifica, decodifica roundtrip */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int bpe_load(const char *path);
int bpe_ntokens(void);
int bpe_encode(const char *text, int *ids, int max_ids);
int bpe_decode(const int *ids, int n, char *out, int outmax);

int main(int argc, char **argv) {
    const char *vocab = (argc >= 2) ? argv[1] : "vocab.bpe";
    int n = bpe_load(vocab);
    if (n < 0) { printf("no carga vocab\n"); return 1; }
    printf("vocab cargado: %d tokens\n", n);

    /* DEBUG: reproduce longest_match de 'the' */
    {
        /* no puedo ver tok_tab static; uso encode de 'th' y 'the' */
        int ids[4];
        bpe_encode("the", ids, 4);
        /* si 'the' fuera token 258, encode daria 1 token */
        printf("('the' == 258?) estudio tokens cargados con 2-3 chars_tmp\n");
    }

    const char *tests[] = {
        "hola que tal",
        "Write a C function that implements a simple arena allocator",
        "la vida es bella y simple, learn each day",
        "function returns the number of bytes which are needed",
        "hola",
    };
    for (int t = 0; t < 5; t++) {
        int ids[512];
        char out[2048];
        int m = bpe_encode(tests[t], ids, 512);
        bpe_decode(ids, m, out, sizeof out);
        int exact = strcmp(tests[t], out) == 0;
        printf("[%s] %zu chars -> %d tokens | roundtrip %s\n",
               exact ? "OK " : "DIF", strlen(tests[t]), m, out[0] ? "" : "");
        if (!exact) printf("    orig : %s\n    deco: %s\n", tests[t], out);
        /* compresion */
        printf("      ratio: %.2f chars/token\n",
               (double)strlen(tests[t]) / (m ? m : 1));
    }
    /* debug: primeros 3 tokens encodeados de 'hola que tal' */
    {
        int ids[64];
        int m = bpe_encode("hola", ids, 64);
        printf("DEBUG 'hola': ");
        for (int i = 0; i < m; i++)
            printf("%d(%s) ", ids[i], (ids[i] < 200 ? "-" : ""));
        printf("\n");
    }
    return 0;
}