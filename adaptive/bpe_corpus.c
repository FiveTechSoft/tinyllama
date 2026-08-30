/* bpe_corpus.c - convierte corpus.bin (bytes+controles) a corpus_ids.bin
 * formato salida: u32 n_ids + u32 ids[] (ids BPE 0..2047)
 * los bytes de control 0x01/0x02/0x03 pasan iguales (son tokens 1,2,3)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int bpe_load(const char *path);
int bpe_encode(const char *text, int *ids, int max_ids);

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "uso: bpe_corpus entrada.bin salida_ids.bin vocab.bpe\n");
        return 1;
    }
    if (bpe_load(argv[3]) < 0) { fprintf(stderr, "no vocab\n"); return 1; }

    FILE *in = fopen(argv[1], "rb");
    if (!in) { fprintf(stderr, "no in\n"); return 1; }
    fseek(in, 0, SEEK_END);
    long sz = ftell(in);
    fseek(in, 0, SEEK_SET);
    uint8_t *data = (uint8_t *)malloc((size_t)sz);
    if (fread(data, 1, (size_t)sz, in) != (size_t)sz) { fclose(in); return 1; }
    fclose(in);

    /* procesa registro a registro: bytes 0x01/0x02/0x03 son tokens propios;
     * entre ellos va texto que se encodea con bpe_encode */
    FILE *out = fopen(argv[2], "wb");
    if (!out) { fprintf(stderr, "no out\n"); return 1; }
    /* escribe placeholder de n_ids y actualiza al final */
    uint32_t placeholder = 0;
    fwrite(&placeholder, 4, 1, out);
    long nids = 0;

    int *ids = (int *)malloc(sizeof(int) * 1 << 20);
    static uint8_t frag[1 << 20];
    int fn = 0;
    for (long i = 0; i < sz; i++) {
        uint8_t b = data[i];
        if (b == 0x01 || b == 0x02 || b == 0x03) {
            /* flush texto pendiente */
            if (fn > 0) {
                int m = bpe_encode((const char *)frag, ids, 1 << 20);
                if (fn < 100)
                    fprintf(stderr, "[sm %d -> %d]\n", fn, m);
                else if (m * 2 < fn)
                    fprintf(stderr, "[dbg reg] frag %d chars -> %d ids\n", fn, m);
                fwrite(ids, 4, (size_t)m, out);
                nids += m;
                fn = 0;
            }
            fwrite(&b, 4, 1, out);
            nids++;
        } else {
            frag[fn++] = b;
        }
    }
    if (fn > 0) {
        int m = bpe_encode((const char *)frag, ids, 1 << 20);
        fwrite(ids, 4, (size_t)m, out);
        nids += m;
    }
    fseek(out, 0, SEEK_SET);
    fwrite(&placeholder /* nids en 4 bytes */, 4, 1, out);
    placeholder = (uint32_t)nids;
    fseek(out, 0, SEEK_SET);
    fwrite(&placeholder, 4, 1, out);
    fclose(out);
    printf("%s: %ld ids (de %ld bytes) = ratio %.2f chars/token\n",
           argv[2], nids, sz, (double)sz / (nids ? nids : 1));
    free((void *)data); free(ids);
    return 0;
}
