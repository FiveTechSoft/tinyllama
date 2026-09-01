/* gpub_test.c - FASE A verificado: ad_gpub_lin exacto vs CPU */
#include "adaptive.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int ad_gpub_init(void);
int ad_gpub_ok(void);
void ad_gpub_free(void);
int ad_gpub_lin(float *out, const float *in, const float *W, const float *bias,
                int BT, int K, int N);

int main(void) {
    if (ad_gpub_init()) { printf("no GPU\n"); return 1; }
    printf("gpub init OK\n");

    /* test: BT=4, K=3, N=2: IN[4x3] @ W[3x2] + bias[2] */
    int BT = 4, K = 3, N = 2;
    float in[12] = {1,2,3, 4,5,6, 7,8,9, 10,11,12};
    float W[6]   = {1,0,1, 0,1,1};        /* row-major 3x2 */
    float bias[2] = {10, 20};
    float out[8];
    if (ad_gpub_lin(out, in, W, bias, BT, K, N)) { printf("lin fail\n"); return 1; }

    /* CPU de referencia */
    int ok = 1;
    for (int i = 0; i < BT; i++) {
        for (int j = 0; j < N; j++) {
            float want = bias[j];
            for (int k = 0; k < K; k++) want += in[i*K+k] * W[k*N+j];
            if (fabsf(out[i*N+j] - want) > 1e-3f) {
                printf("MISMATCH [%d][%d]: gpu %g vs cpu %g\n",
                       i, j, out[i*N+j], want);
                ok = 0;
            }
        }
    }
    printf("FASE A: gpub_lin %s\n", ok ? "OK (exacto vs CPU)" : "FALLO");
    ad_gpub_free();
    return ok ? 0 : 1;
}