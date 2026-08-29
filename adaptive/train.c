/* train.c - entrenador del LLM adaptativo: backprop completo + Adam + gate
 *
 * Entrena CE del siguiente byte sobre ventanas aleatorias del corpus.
 * Solo escribe --out si la val-PPL mejora frente al modelo de entrada.
 *
 * Uso:
 *   train.exe --fresh 96,4,4 --data corpus.bin [--steps 2000] [--out m.adm]
 *   train.exe --model in.adm --data corpus.bin --steps 3000 [--val val.bin]
 *   opciones: --lr 3e-4 --batch 8
 */
#include "ad_train.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static int   OPT_STEPS = 2000;
static int   OPT_T     = 96;
static float OPT_LR    = 3e-4f;

/* buffers de trabajo de la ventana en curso */
static float *wx, *wh, *wqkv, *wao, *wproj, *wl2o, *wf1, *wf1a, *wlog;
static float *wdx, *wdh, *wdqkv, *wdao, *wdp, *wscore;
static float *grads;               /* arena de gradientes = ad_total_floats */
static float *mad, *vad;           /* Adam */
static int wcap = 0;
static AdModel *M = NULL;

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "OOM %zu\n", n); exit(1); }
    return p;
}

static uint8_t *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return NULL; }
    uint8_t *b = (uint8_t *)malloc((size_t)sz);
    if (!b) { fclose(f); return NULL; }
    if (fread(b, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(b); return NULL; }
    fclose(f);
    *len = (size_t)sz;
    return b;
}

/* backward de un linear: dW[r,c] += dy[r]*x[c] */
static void grad_W(float *dW, const float *dy, const float *x, int R, int C) {
    for (int r = 0; r < R; r++) {
        float *dwr = dW + (size_t)r * C;
        float yv = dy[r];
        for (int c = 0; c < C; c++) dwr[c] += yv * x[c];
    }
}

/* ============ forward de una ventana (full seq, eval) ============= */
static float window_eval(AdModel *m, const uint8_t *bytes, int T) {
    const int dim = m->cfg.dim, hd = m->cfg.head_dim, nh = m->cfg.n_heads;
    const int hid = m->cfg.hidden;
    const size_t d = (size_t)dim;
    float loss = 0.f;
    /* buffers (estaticos, crecen a demanda) */
    static float *xs, *hs, *qkvs, *aots, *ps, *f1s, *lg; static int cap = 0;
    static float *kcs, *vcs;
    if (T > cap) {
        free(xs); free(h); free(qkv); free(aots); free(f1s); free(logs);
        xs  = (float *)calloc((size_t)T, dim * sizeof(float));
        ...
    }
    return 0.f;
}

int main(int argc, char **argv) {
    printf("build\n");
    return 0;
}