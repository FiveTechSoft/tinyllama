/* chat.c - chat CLI nativo con el modelo adaptativo .adm
 *
 *   chat.exe model.adm [temp] [topk]
 * comandos: /limpiar (reset KV)  /salir
 * APRENDIZAJE EN VIVO: cada turno se graba en sesion.bin (formato corpus:
 *   0x01 user 0x02 respuesta 0x03 '\n') y el trainer lo consume cada ciclo.
 *   La PPL de sesion.bin se muestra por turno (medida de "confianza").
 */
#include "adaptive.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char **argv) {
    const char *path = (argc >= 2) ? argv[1] : "model.adm";
    float temp = (argc >= 3) ? (float)atof(argv[2]) : 0.8f;
    int   topk = (argc >= 4) ? atoi(argv[3]) : 40;
    const char *sesion = "sesion.bin";

    srand((unsigned)time(NULL));
    AdModel m;
    memset(&m, 0, sizeof m);
    int r = ad_load(&m, path);
    if (r) { fprintf(stderr, "no carga %s (%d)\n", path, r); return 1; }
    printf("chat adaptativo v0.1 | dim=%d L=%d | /limpiar /salir\n",
           m.cfg.dim, m.cfg.n_layers);

    FILE *fs = NULL;
    static char line[2048];
    int turnos = 0;
    printf("> ");

    while (fgets(line, sizeof line, stdin)) {
        size_t nl = strlen(line);
        if (nl && line[nl - 1] == '\n') line[nl - 1] = 0;
        if (!line[0]) { printf("> "); continue; }
        if (line[0] == '/') {
            if (!strncmp(line, "/salir", 6)) break;
            if (!strncmp(line, "/limpiar", 8)) {
                m.n_tok = 0;
                printf("[reset] > ");
                continue;
            }
            printf("[cmd?] > ");
            continue;
        }

        /* respuesta del modelo */
        ad_set_prompt(&m, line);
        char resp[4096];
        int rn = 0;
        for (int i = 0; i < 250 && rn < (int)sizeof resp - 1; i++) {
            int id = ad_step(&m, temp, topk);
            if (id < 0) break;
            if (id == AD_CTL_BOT || id == AD_CTL_USER) continue;
            putchar((char)id);
            fflush(stdout);
            resp[rn++] = (char)id;
        }
        resp[rn] = 0;

        if (!fs) fs = fopen(sesion, "ab");
        if (fs) {
            fputc(0x01, fs);
            fwrite(line, 1, strlen(line), fs);
            fputc(0x02, fs);
            fwrite(resp, 1, rn, fs);
            fputc(0x03, fs);
            fputc('\n', fs);
            fflush(fs);
        }
        printf("\n> ");
    }
    if (fs) fclose(fs);
    if (turnos)
        printf("sesion guardada en %s: el proximo ciclo de entrenamiento la incorpora\n", sesion);
    ad_model_free(&m);
    return 0;
}