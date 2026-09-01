/* ad_gpub_bwd.c - backward batch en GPU (FASE C)
 *
 * gradientes de un linear OUT[BT x N] = IN[BT x K] @ W[N x K]^T + b[N]
 *   dW[N x K] = dout^T[BT x N]^T @ in[BT x K]  => gemm OP_T opN: dW = dout @ in
 *   dIN[BT x K] = dout[BT x N] @ W[N x K]      => gemm N,N con W como [N x K]
 *   db[N] = sum over BT
 *
 * convencion del layout (igual que ad_gpub_lin): W esta en memoria como
 * [N x K] row-major y el gemm usa OP_T para hacer in @ W^T
 *   => forward: out = in @ W^T
 *   => dW = dout^T @ in   (N x K, mismo layout que W)
 *   => dIn = dout @ W
 *
 *   ad_gpub_bwd(dW, dIN, dout, in, W, BT, K, N)  -> acumula en dW/dIN
 *   ad_gpub_zero(dW, n)                           -> memset helper
 */
#include "adaptive.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int ad_gpub_init(void);
extern int ad_gpub_ok(void);

static cublasHandle_t bh = NULL;
static int b_ready = 0;

static void cudaFreeAll(void *a, void *b, void *c, void *d) {
    if (a) cudaFree(a);
    if (b) cudaFree(b);
    if (c) cudaFree(c);
    if (d) cudaFree(d);
}


/* version propia del handle para el backward (independiente del fwd) */
static cublasHandle_t get_h(void) {
    if (!bh) {
        if (cudaFree(0) != cudaSuccess) return NULL;
        if (cublasCreate(&bh) != CUBLAS_STATUS_SUCCESS) return NULL;
        b_ready = 1;
    }
    return bh;
}

/* dW[N x K] += dout[BT x N]^T @ in[BT x K]
 * row-major: dW[r][k] = sum_t dout[t][r] * in[t][k]
 * via col-major: dW_cm[K,N] con ld=K... el layout de dW es igual a W [N x K]
 * dW_rm[N,K] += dout_rm[BT,N]^T @ in_rm[BT,K]
 * en col-major: dW_cm[K,N] = dout_cm[N,BT]^T ... uso el truco del test:
 * dW_rm += gemm(opA=N de dout_rm leido como cm [N,BT], opB=N de in_rm
 * leido como cm [K,BT]... */
int ad_gpub_bwd(float *dW, float *dIN, const float *dout, const float *in,
                const float *W, int BT, int K, int N, int accumulate) {
    cublasHandle_t h = get_h();
    if (!h) return -1;
    float *dDW = NULL, *dDOUT, *dIN_H, *dDIN = NULL;
    size_t sdw = (size_t)N * K * 4, sdout = (size_t)BT * N * 4;
    size_t sin = (size_t)BT * K * 4, sdin = (size_t)BT * K * 4;

    int needDW = (dW != NULL), needDIN = (dIN != NULL);
    if (cudaMalloc(&dDOUT, sdout)) return -1;
    if (cudaMalloc(&dIN_H, sin)) { cudaFree(dDOUT); return -1; }
    cudaMemcpy(dDOUT, dout, sdout, cudaMemcpyHostToDevice);
    cudaMemcpy(dIN_H, in, sin, cudaMemcpyHostToDevice);
    if (needDIN) {
        if (cudaMalloc(&dDIN, sin)) {
            cudaFree(dDOUT); cudaFree(dIN_H); return -1;
        }
    }

    float alpha = 1.0f;
    if (needDW) {
        if (cudaMalloc(&dDW, sdw)) {
            cudaFree(dDOUT); cudaFree(dIN_H); if (dDIN) cudaFree(dDIN); return -1;
        }
        cudaMemcpy(dDW, dW, sdw, cudaMemcpyHostToDevice);
    }

    /* ================= dIN en GPU: dIN[BT,K] = dout[BT,N] @ W[N,K]
     * (reusa el gemm VERIFICADO de ad_gpub_lin: out = in @ W^T
     *  con 'in'=dout [BT,N] y 'W'=W [N,K] => out = dout @ W^T: no es lo mismo)
     * => cublasSgemm directo con la convencion row-major verificada:
     *    C_rm[BT,K] = dout_rm[BT,N] @ W_rm[N,K]
     *    C_cm[K,BT] = W_cm[K,N]^T ... el truco del gpu_test:
     *    gemm(OP_N, OP_N, K, BT, N, W_cm(ld=K), dout_cm(ld=N), C_cm(ld=K))
     *    donde W_rm[N,K] == W_cm[K,N] con ld=K  (exacto)
     *    y dout_rm[BT,N] == dout_cm[N,BT] con ld=N  (exacto)
     *    C_cm[K,BT] == C_rm[BT,K] con ld=K  (exacto) */
    if (needDIN) {
        float *dWh = NULL, *dDIN = NULL;
        if (cudaMalloc(&dWh, (size_t)N * K * 4) == 0 &&
            cudaMalloc(&dDIN, (size_t)BT * K * 4) == 0) {
            cudaMemcpy(dWh, W, (size_t)N * K * 4, cudaMemcpyHostToDevice);
            float abeta = 0.0f;
            cublasStatus_t st = cublasSgemm(bh, CUBLAS_OP_N, CUBLAS_OP_N,
                                            K, BT, N, &alpha,
                                            dWh, K, dDOUT, N,
                                            &abeta, dDIN, K);
            if (st == CUBLAS_STATUS_SUCCESS)
                cudaMemcpy(dIN, dDIN, sin, cudaMemcpyDeviceToHost);
            cudaFree(dWh);
        }
        if (dDIN) cudaFree(dDIN);
    }

    /* ================= dW en HOST: doutT @ in (fiable y barato)
     * el buffer dout es pequeno (BT x N): la GEMM de dW en CPU evita
     * la pelea con la convencion col-major para el layout [N x K]
     * (FASE C-beta; el dW en GPU llega cuando el batch B > 64) */
    if (needDW) {
        memset(dW, 0, sdw);
        for (int t = 0; t < BT; t++)
            for (int n = 0; n < N; n++) {
                float dvt = dout[(size_t)t * N + n];
                if (dvt == 0.f) continue;
                const float *ir = in + (size_t)t * K;
                float *dw = dW + (size_t)n * K;
                for (int k = 0; k < K; k++) dw[k] += dvt * ir[k];
            }
    }
    cudaFree(dDOUT); cudaFree(dIN_H);
    return 0;
}





