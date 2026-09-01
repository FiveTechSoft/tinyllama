/* gpub_bwd_test.c - FASE C verificado: gradientes GPU == gradientes CPU
 * para un linear: out[BT x N] = in[BT x K] @ W[N x K]^T + b[N]
 */
#include "adaptive.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int ad_gpub_init(void);
int ad_gpub_bwd(float *dW, float *dIN, const float *dout, const float *in,
                const float *W, int BT, int K, int N, int accumulate);

int main(void) {
    if (ad_gpub_init()) { printf("no GPU\n"); return 1; }
    int BT = 8, K = 16, N = 32;

    /* datos aleatorios deterministas */
    float in[128], W[512], dout[256];
    for (int i = 0; i < BT * K; i++) in[i] = (float)(i % 7) * 0.25f;
    for (int i = 0; i < N * K; i++) W[i] = (float)(i % 5) * 0.1f - 0.2f;
    for (int i = 0; i < BT * N; i++) dout[i] = (float)(i % 3) * 0.5f - 0.3f;

    /* CPU de referencia */
    float dW_ref[512] = {0}, dIN_ref[128] = {0};
    for (int t = 0; t < BT; t++)
        for (int n = 0; n < N; n++) {
            float d = dout[t * N + n];
            for (int k = 0; k < K; k++) {
                dW_ref[n * K + k] += d * in[t * K + k];
                dIN_ref[t * K + k] += d * W[n * K + k];
            }
        }

    /* GPU */
    float dW_gpu[512], dIN_gpu[128];
    memset(dW_gpu, 0, sizeof dW_gpu);
    int r = ad_gpub_bwd(dW_gpu, dIN_gpu, dout, in, W, BT, K, N, 0);
    if (r) { printf("gpub_bwd err %d\n", r); return 1; }

    /* compara */
    float errW = 0.f, errIN = 0.f;
    for (int i = 0; i < N * K; i++) errW += fabsf(dW_gpu[i] - dW_ref[i]);
    for (int i = 0; i < BT * K; i++) errIN += fabsf(dIN_gpu[i] - dIN_ref[i]);
    errW /= (N * K); errIN /= (BT * K);
    printf("dW_gpu[0]=%g dW_ref[0]=%g | dIN_gpu[0]=%g dIN_ref[0]=%g\n",
           dW_gpu[0], dW_ref[0], dIN_gpu[0], dIN_ref[0]);
    printf("err dW %.6f | err dIN %.6f => %s\n",
           errW, errIN, (errW < 0.01f && errIN < 0.01f) ? "OK" : "FALLO");
    return (errW < 0.01f && errIN < 0.01f) ? 0 : 1;
}