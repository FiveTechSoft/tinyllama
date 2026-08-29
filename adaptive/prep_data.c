/* prep_data.c - JSONL de peng-mimo -> corpus.bin
 * entrada: ficheros .jsonl con {"id":...,"prompt":"P","tags":[...],"response":"R"}
 * salida:  por registro -> byte 0x01 + P + 0x02 + R + 0x03 + '\n'
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* devuelve puntero al primer char del valor (tras la comilla) o NULL */
static const char *find_key(const char *line, const char *key) {
    char pat[80];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(line, pat);
    if (!p) return NULL;
    p += strlen(pat);
    while (*p == ' ') p++;
    if (*p != ':') return NULL;
    p++;
    while (*p == ' ') p++;
    if (*p != '"') return NULL;
    return p + 1;
}

/* copia hasta comilla de cierre; \n \t \" \\ decodificados; \uNNNN basico */
static size_t copy_str(const char *pv, char *out, size_t outsz) {
    size_t n = 0;
    while (*pv && *pv != '"') {
        if (*pv == '\\' && pv[1]) {
            pv++;
            char c = *pv;
            if (c == 'n') out[n++] = '\n';
            else if (c == 't') out[n++] = '\t';
            else if (c == '"') out[n++] = '"';
            else if (c == '\\') out[n++] = '\\';
            else if (c == 'u') {
                unsigned cp = 0;
                for (int k = 1; k <= 4 && pv[k]; k++) {
                    char h = pv[k];
                    unsigned v = (h >= '0' && h <= '9') ? (unsigned)(h - '0')
                        : (h >= 'a' && h <= 'f') ? (unsigned)(h - 'a' + 10)
                        : (h >= 'A' && h <= 'F') ? (unsigned)(h - 'A' + 10) : 0;
                    cp = cp * 16 + v;
                }
                if (cp < 0x80) {
                    out[n++] = (char)cp;
                } else if (cp < 0x800) {
                    out[n++] = (char)(0xC0 | (cp >> 6));
                    out[n++] = (char)(0x80 | (cp & 0x3F));
                } else {
                    out[n++] = (char)(0xE0 | (cp >> 12));
                    out[n++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                    out[n++] = (char)(0x80 | (cp & 0x3F));
                }
                pv += 4;
            } else {
                out[n++] = c;
            }
            pv++;
        } else {
            out[n++] = *pv++;
        }
        if (n >= outsz - 4) break;
    }
    out[n] = 0;
    return n;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "uso: prep_data salida.bin in1.jsonl [in2...]\n");
        return 1;
    }
    FILE *out = fopen(argv[1], "wb");
    if (!out) { fprintf(stderr, "no salida %s\n", argv[1]); return 1; }

    static char line[262144];
    static char prompt[65536];
    static char resp[131072];
    size_t nrec = 0, nskip = 0;

    for (int a = 2; a < argc; a++) {
        FILE *in = fopen(argv[a], "rb");
        if (!in) { fprintf(stderr, "no abre %s\n", argv[a]); continue; }
        while (fgets(line, sizeof line, in)) {
            const char *pv = find_key(line, "prompt");
            const char *rv = find_key(line, "response");
            if (!pv || !rv) { nskip++; continue; }
            size_t plen = copy_str(pv, prompt, sizeof prompt);
            size_t rlen = copy_str(rv, resp, sizeof resp);
            if (!plen || !rlen) { nskip++; continue; }
            fputc(0x01, out);
            fwrite(prompt, 1, plen, out);
            fputc(0x02, out);
            fwrite(resp, 1, rlen, out);
            fputc(0x03, out);
            fputc('\n', out);
            nrec++;
        }
        fclose(in);
    }
    fclose(out);
    printf("%zu registros OK, %zu saltados -> %s\n", nrec, nskip, argv[1]);
    return 0;
}