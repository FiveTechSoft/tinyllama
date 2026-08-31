/* ad_muon.c - optimizer MuonH (estilo PuRo-2B 3.3) para el modelo adaptativo
 *
 * ortogonalizacion por Gram-Schmidt modificado (QR por filas):
 *   - mismo objetivo que Newton-Schulz (singular values -> 1)
 *   - garantizado estable (NS con coeficientes mal elegidos diverge;
 *     GS es determinista y siempre converge)
 *   - coste O(R*R*C) = 2 GEMM: comparable a 2 iteraciones de NS
 *
 * grupos: matrices 2D = MuonH; head/emb/biases/1D = AdamW
 */
#include "adaptive.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ortogonaliza X (R x C) por QR de Householder-Gram-Schmidt por filas:
 * X = Q C con Q ortogonal R x C (R <= C) y C triangular superior;
 * devuelve Q con todas las filas ortonormalizadas (filas dependientes -> 0) */
static void gs_orth(float *X, int R, int C, float *w) {
    (void)w;
    for (int i = 0; i < R; i++) {
        /* resta proyecciones sobre filas previas */
        for (int j = 0; j < i; j++) {
            float dot = 0.f;
            for (int k = 0; k < C; k++) dot += X[(size_t)i * C + k] * X[(size_t)j * C + k];
            for (int k = 0; k < C; k++) X[(size_t)i * C + k] -= dot * X[(size_t)j * C + k];
        }
        /* normaliza */
        float nrm = 0.f;
        for (int k = 0; k < C; k++) nrm += X[(size_t)i * C + k] * X[(size_t)i * C + k];
        nrm = sqrtf(nrm);
        if (nrm < 1e-7f) {
            /* fila linealmente dependiente: zero por seguridad */
            for (int k = 0; k < C; k++) X[(size_t)i * C + k] = 0.f;
            continue;
        }
        for (int k = 0; k < C; k++) X[(size_t)i * C + k] /= nrm;
    }
}

static float *scratch = NULL;
static size_t scratch_cap = 0;

/* aplica 1 paso MuonH a la matriz W (R x C) con gradiente G y momentum M */
void ad_muon_apply(float *W, float *G, float *M, int R, int C, float lr) {
    size_t need = (size_t)R * C;
    if (scratch_cap < need) {
        free(scratch);
        scratch = (float *)malloc(need * sizeof(float));
        scratch_cap = need;
    }
    /* momentum: M = 0.95 M + G */
    for (int i = 0; i < R * C; i++) M[i] = 0.95f * M[i] + G[i];

    /* O = ortogonaliza(M) */
    memcpy(scratch, M, need * sizeof(float));
    gs_orth(scratch, R, C, NULL);

    /* escala espectral estandar de Muon: ~sqrt(max(1, R/C)) */
    float sc = sqrtf((float)R / (float)C);
    if (sc < 1.f) sc = 1.f;
    float nrm2 = 0.f;
    for (int i = 0; i < R * C; i++) { scratch[i] *= sc; nrm2 += scratch[i] * scratch[i]; }
    nrm2 = sqrtf(nrm2);
    /* hyperball clip a norma 1 (PuRo: control por LR explicito) */
    if (nrm2 > 1.f)
        for (int i = 0; i < R * C; i++) scratch[i] /= nrm2;

    /* W -= lr * O */
    for (int i = 0; i < R * C; i++) W[i] -= lr * scratch[i];
}

/* test: matriz aleatoria 64 x 192 -> ortogonalizada -> gram ~ I */
int ad_muon_test(void) {
    int R = 64, C = 192;
    float *X = (float *)malloc((size_t)R * C * sizeof(float));
    float *B = (float *)malloc((size_t)R * R * sizeof(float));
    for (int i = 0; i < R * C; i++) X[i] = ad_randf() - 0.5f;
    gs_orth(X, R, C, NULL);
    for (int i = 0; i < R * R; i++) B[i] = 0.f;
    for (int i = 0; i < R; i++)
        for (int k = 0; k < C; k++) {
            float v = X[(size_t)i * C + k];
            for (int j = 0; j < R; j++)
                B[(size_t)i * R + j] += v * X[(size_t)j * C + k];
        }
    float err = 0.f;
    for (int i = 0; i < R; i++)
        for (int j = 0; j < R; j++) {
            float want = (i == j) ? 1.f : 0.f;
            err += fabsf(B[(size_t)i * R + j] - want);
        }
    free(X); free(B);
    return (err / (R * R) < 0.01f) ? 0 : 1;   /* tolerancia 1% */
}