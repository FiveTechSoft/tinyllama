/* bpe.c - tokenizador BPE para el LLM adaptativo
 *
 * vocab.bpe (commiteado): una linea por merge "id<TAB>strA<TAB>strB",
 * ids 256+ en orden. ids 0-255 = bytes, excepto bytes de control 0x01,0x02,0x03
 * que tokenizan tal cual.
 *
 *   bpe_load(path)                    -> carga vocab (0 ok)
 *   bpe_encode(text, ids, max)        -> texto -> ids (n devueltos)
 *   bpe_decode(ids, n, out, outmax)   -> ids -> texto
 *
 * encoder: greedy longest-match sobre la tabla de tokens (los merges
 * construyen la tabla completa id->string; ordenamos por longitud y
 * escaneamos). no es BPE-min-pairs exacto del trainer, pero es
 * determinista y consistente con el vocab aprendido.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define BPE_MAX_TOKENS 2048

typedef struct {
    char str[64];       /* texto del token (UTF-8) */
    int  str_len;
    int  a, b;          /* pares del merge (o -1 para bytes) */
} BpeTok;

static BpeTok tok_tab[BPE_MAX_TOKENS];
static int ntok = 256;

/* por defecto, token i = char(i) para i<256 */
static int loaded = 0;

int bpe_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char line[512];
    for (int i = 0; i < 256; i++) {
        tok_tab[i].str[0] = (char)i;
        tok_tab[i].str[1] = 0;
        tok_tab[i].str_len = 1;
        tok_tab[i].a = tok_tab[i].b = -1;
    }
    ntok = 256;
    int loaded_merges = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#') continue;
        /* parse manual: id TAB sa TAB sb (sa/sb pueden tener espacios) */
        char *endp = NULL;
        long id = strtol(line, &endp, 10);
        if (id < 256 || id >= BPE_MAX_TOKENS || !endp || *endp != '\t') continue;
        char *sa = endp + 1;
        char *tab2 = strchr(sa, '\t');
        if (!tab2) continue;
        *tab2 = 0;
        char *sb = tab2 + 1;
        char *nl = strchr(sb, '\n');
        if (nl) *nl = 0;
        /* Windows: fgets trae \r\n -> recorta \r final y cualquier \r suelto */
        for (char *z = sb; *z; z++)
            if (*z == '\r') { *z = 0; break; }
        char *cr = strchr(sa, '\r');
        if (cr) *cr = 0;
        BpeTok *t = &tok_tab[id];
        snprintf(t->str, sizeof t->str, "%s%s", sa, sb);
        t->str_len = (int)strlen(t->str);
        t->a = (int)id; t->b = 0;
        ntok = id + 1;
        loaded_merges++;
    }
    fclose(f);
    loaded = 1;
    return ntok;
}

int bpe_ntokens(void) { return loaded ? ntok : 256; }

/* longitud del match mas largo que empiece en text[pos]
 * buceo por buckets de primer byte (O(candidatos) no O(2048)) */
typedef struct { int ids[256]; int n; } Bucket;
static Bucket buckets[256];      /* indexado por primer byte del token */
static int buckets_ok = 0;

static int longest_match(const unsigned char *text, int pos, int len) {
    if (!buckets_ok) { /* lazily construye una vez */
        for (int i = 0; i < ntok; i++) {
            if (tok_tab[i].str_len < 1) continue;
            unsigned char c = (unsigned char)tok_tab[i].str[0];
            if (buckets[c].n < 256) buckets[c].ids[buckets[c].n++] = i;
        }
        buckets_ok = 1;
    }
    int best = -1;
    int bestlen = 0;
    unsigned char c0 = text[pos];
    Bucket *bk = &buckets[c0];
    for (int k = 0; k < bk->n; k++) {
        int i = bk->ids[k];
        int L = tok_tab[i].str_len;
        if (L <= bestlen || L > len - pos) continue;
        if (memcmp(text + pos, tok_tab[i].str, (size_t)L) == 0) {
            best = i; bestlen = L;
        }
    }
    return best;
}

/* greedy longest match: texto -> ids */
int bpe_encode(const char *text, int *ids, int max_ids) {
    const unsigned char *p = (const unsigned char *)text;
    int len = (int)strlen(text);
    int n = 0;
    int i = 0;
    while (i < len && n < max_ids) {
        int t = longest_match(p, i, len);
        if (t >= 0) {
            ids[n++] = t;
            i += tok_tab[t].str_len;
        } else {
            ids[n++] = p[i];  /* byte suelto */
            i++;
        }
    }
    return n;
}

int bpe_decode(const int *ids, int n, char *out, int outmax) {
    int o = 0;
    for (int i = 0; i < n && o < outmax - 1; i++) {
        int id = ids[i];
        if (id < 0 || id >= ntok) continue;
        int L = tok_tab[id].str_len;
        if (o + L >= outmax) break;
        memcpy(out + o, tok_tab[id].str, (size_t)L);
        o += L;
    }
    out[o] = 0;
    return o;
}