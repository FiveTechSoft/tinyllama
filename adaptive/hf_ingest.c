/*
 * hf_ingest.c - descarga y convierte datasets de HF a corpus.bin
 * Fuente elegida (instrucciones, calidad, tamano razonable):
 *   1. yahma/alpaca-cleaned (52k instrucciones EN, chat SFT clasico)
 *   2. roneneldan/TinyStories (cuentos simples, gramatica consistente)
 * Descarga via URL directa (resolve/main) sin libs externas (pure C + HTTP).
 * Para no reinventar HTTPS en C, genera un script .py que ya corre en el
 * workflow; este binario solo convierte los JSONL descargados a corpus.
 *
 * Conversion:
 *   alpaca: {"instruction","input","output"} -> 0x01 inst+input 0x02 output 0x03
 *   tinystories JSONL {"story"}             -> texto plano entre control EOS
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static size_t copy_json_str(const char *pv, char *out, size_t outsz) {
    size_t n = 0;
    while (*pv && *pv != '"') {
        if (*pv == '\\' && pv[1]) {
            pv++;
            char c = *pv;
            if (c == 'n') out[n++] = '\n';
            else if (c == 't') out[n++] = '\t';
            else if (c == '"') out[n++] = '"';
            else if (c == '\\') out[n++] = '\\';
            else if (c == '/') out[n++] = '/';
            else if (c == 'u') {
                unsigned cp = 0;
                for (int k = 1; k <= 4 && pv[k]; k++) {
                    char h = pv[k];
                    unsigned v = (h >= '0' && h <= '9') ? (unsigned)(h - '0')
                               : (h >= 'a' && h <= 'f') ? (unsigned)(h - 'a' + 10)
                               : (h >= 'A' && h <= 'F') ? (unsigned)(h - 'A' + 10) : 0;
                    cp = cp * 16 + v;
                }
                if (cp < 0x80) out[n++] = (char)cp;
                else if (cp < 0x800) {
                    out[n++] = (char)(0xC0 | (cp >> 6));
                    out[n++] = (char)(0x80 | (cp & 0x3F));
                } else {
                    out[n++] = (char)(0xE0 | (cp >> 12));
                    out[n++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                    out[n++] = (char)(0x80 | (cp & 0x3F));
                }
                pv += 4;
            } else out[n++] = c;
            pv++;
        } else {
            out[n++] = *pv++;
        }
        if (n >= outsz - 4) break;
    }
    out[n] = 0;
    return n;
}

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

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "uso: hf_ingest salida.bin tipo in1.jsonl [in2...]\n"
                        "  tipo: alpaca | stories\n");
        return 1;
    }
    const char *tipo = argv[2];
    FILE *out = fopen(argv[1], "wb");
    if (!out) { fprintf(stderr, "no salida\n"); return 1; }

    static char line[1 << 20];
    static char a[131072], b[131072];
    size_t nrec = 0, nskip = 0;

    for (int f = 3; f < argc; f++) {
        FILE *in = fopen(argv[f], "rb");
        if (!in) { fprintf(stderr, "no abre %s\n", argv[f]); continue; }
        while (fgets(line, sizeof line, in)) {
            if (!strncmp(tipo, "alpaca", 6)) {
                const char *iv = find_key(line, "instruction");
                if (!iv) continue;
                size_t ilen = copy_json_str(iv, a, sizeof a);
                /* input opcional */
                const char *ipv = find_key(line, "input");
                size_t plen = 0;
                if (ipv) {
                    size_t xl = copy_json_str(ipv, b, sizeof b);
                    if (xl && xl < sizeof a - ilen - 4) {
                        ilen += snprintf(a + ilen, sizeof a - ilen, ": %s", b);
                    }
                }
                const char *ov = find_key(line, "output");
                if (!ov) { nskip++; continue; }
                size_t olen = copy_json_str(ov, b, sizeof b);
                if (!ilen || !olen) { nskip++; continue; }
                fputc(0x01, out);
                fwrite(a, 1, ilen, out);
                fputc(0x02, out);
                fwrite(b, 1, olen, out);
                fputc(0x03, out);
                fputc('\n', out);
                nrec++;
            } else { /* stories: texto crudo */
                /* TinyStories JSONL: {"story": "..."} */
                const char *sv = find_key(line, "story");
                if (!sv) continue;
                size_t slen = copy_json_str(sv, a, sizeof a);
                if (!slen) { nskip++; continue; }
                fputc(0x01, out);
                fwrite(a, 1, slen, out);
                fputc(0x03, out);
                fputc('\n', out);
                nrec++;
            }
        }
        fclose(in);
    }
    fclose(out);
    printf("%zu registros, %zu saltados -> %s\n", nrec, nskip, argv[1]);
    return 0;
}