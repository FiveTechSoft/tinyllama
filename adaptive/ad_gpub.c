/* ad_gpub.c - backend GPU batch para el trainer (cuBLAS)
 *
 * CONVENCION DOCUMENTADA (la unica que cuadra sin pelear):
 *   - en el host, TODO es row-major
 *   - para cublas (col-major) usamos el truco de la transposicion
 *     implicita: C_rm[R,C] = A_rm[R,K] @ B_rm[K,C]
 *     es igual a: cublasSgemm(OP_T, OP_N, C, R, K, B_rm, ldb=C, A_rm, ldb=K)
 *     verificado por gpu_test.c (matmul exacto [58 64 139 154])
 *
 *   ad_gpub_lin(out, in, W, bias, BT, K, N):
 *     out[BT x N] = in[BT x K] @ W[K x N] + bias[N]
 *     => equivale a: A=in[BT x K] (op B), B=W[K x N] (op T): m=N n=BT k=K
 *     cublasSgemm(OP_T, OP_N, N, BT, K, alpha, dW, ldb=K, dIN, ldb=K...)
 *     W row-major [K,N] como col-major [N,K] con ld=N => OP_T sobre esa
 *     da [K,N]: C_cm[N,BT] = (W_cm^T [K,N]) @ IN_cm[K,BT]... LA COMBINACION:
 *     m=N, n=BT, k=K, opA=T con A=dW ld=N, opB=N con B=dIN ld=K
 */
#include "adaptive.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

static cublasHandle_t gh = NULL;
static int gpu_ready = 0;

int ad_gpub_init(void) {
    if (cudaFree(0) != cudaSuccess) return -1;
    if (cublasCreate(&gh) != CUBLAS_STATUS_SUCCESS) return -2;
    gpu_ready = 1;
    return 0;
}

int ad_gpub_ok(void) { return gpu_ready; }
void ad_gpub_free(void) {
    if (gh) { cublasDestroy(gh); gh = NULL; gpu_ready = 0; }
}

/* OUT [BT x N] = IN [BT x K] @ W [K x N] + bias[N]
 * cublas (col-major): usamos C = A^T * B^T con A=W, B=IN:
 *   C_rm[R,C] = A_rm[R,K] @ B_rm[K,C]
 *   C_cm[C,R] = B_cm[C,K] @ A_cm[K,R]
 * con A_rm[R,K] en memoria == A_cm[K,R] (ld=K) y B_rm[K,C] == B_cm[C,K] (ld=C)
 * => gemm(opA=N, opB=N, m=C, n=R, k=K, A=B_rm(ldb=C), B=A_rm(ldb=K), C=C_cm ldc=C)
 * (esto es EXACTAMENTE lo que gpu_test.c verifico exacto) */
int ad_gpub_lin(float *out, const float *in, const float *W, const float *bias,
                int BT, int K, int N) {
    if (!gpu_ready) return -1;
    float *dIn, *dW, *dOut;
    size_t sin = (size_t)BT * K * 4, sw = (size_t)K * N * 4, so = (size_t)BT * N * 4;
    if (cudaMalloc(&dIn, sin)) return -1;
    if (cudaMalloc(&dW, sw))  { cudaFree(dIn); return -1; }
    if (cudaMalloc(&dOut, so)){ cudaFree(dIn); cudaFree(dW); return -1; }
    cudaMemcpy(dIn, in, sin, cudaMemcpyHostToDevice);
    cudaMemcpy(dW, W, sw, cudaMemcpyHostToDevice);

    float alpha = 1.0f, beta = 0.0f;
    /* in y W estan intercambiados a proposito: in es el B del gemm
     * (row-major in[BT,K] == col-major in_cm[K,BT] con ld=K)
     * W es el A del gemm (row-major W[K,N] == col-major W_cm[N,K] con ld=N)
     * out row-major [BT,N] == col-major out_cm[N,BT] (ld=N) */
    cublasStatus_t st = cublasSgemm(gh, CUBLAS_OP_N, CUBLAS_OP_N,
                                    N, BT, K, &alpha,
                                    dW, N,   /* W_cm [N x K], ld = N */
                                    dIn, K,  /* IN_cm [K x BT], ld = K */
                                    &beta, dOut, N);
    if (st != CUBLAS_STATUS_SUCCESS) {
        cudaFree(dIn); cudaFree(dW); cudaFree(dOut);
        return -2;
    }
    /* bias en host (N es pequeno): bajo, sumo, subo */
    if (bias) {
        float *tmp = (float *)malloc(so);
        cudaMemcpy(tmp, dOut, so, cudaMemcpyDeviceToHost);
        for (int i = 0; i < BT; i++)
            for (int j = 0; j < N; j++) tmp[(size_t)i * N + j] += bias[j];
        cudaMemcpy(dOut, tmp, so, cudaMemcpyHostToDevice);
        free(tmp);
    }
    cudaMemcpy(out, dOut, so, cudaMemcpyDeviceToHost);
    cudaFree(dIn); cudaFree(dW); cudaFree(dOut);
    return 0;
}