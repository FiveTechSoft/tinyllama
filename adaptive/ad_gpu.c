/* ad_gpu.c - backend CUDA/cuBLAS para el modelo adaptativo
 *
 * estrategia: el forward/backward secuencial por capas queda en CPU
 * (los bucles son chicos), pero las GEMM dominantes (matmul de las
 * matrices Wqkv/Wproj/W1/W2/Whead y sus gradientes) van a GPU via cuBLAS.
 * batch de ventanas grande: los tensores suben a VRAM, se ejecutan los
 * kernels, y bajan solo logits/loss por step.
 *
 * API:
 *   gpu_init()                          -> cublas handle (0 ok)
 *   gpu_matmul(out, A, B, R, K, C)      -> out[R x C] = A[R x K] @ B[K x C]
 *   gpu_free()
 *
 * compilacion (Windows):
 *   nvcc -O3 -o train_gpu.exe train.c ad_core.c bpe.c ad_gpu.c ^
 *        -lcublas -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include"
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

static cublasHandle_t hnd = NULL;
static int gpu_ok = 0;

int gpu_init(void) {
    cudaError_t e = cudaFree(0);
    if (e != cudaSuccess) return -1;
    if (cublasCreate(&hnd) != CUBLAS_STATUS_SUCCESS) return -2;
    gpu_ok = 1;
    return 0;
}

int gpu_available(void) { return gpu_ok; }

/* out[R x C] row-major = A[R x K] @ B[K x C]
 * cuBLAS es column-major: usamos el truco de transponer indices:
 * out^T = B^T @ A^T -> llamamos gemm con op(A)=T
 * row-major R x C == col-major C x R: out_cm = B_cm^T @ A_cm^T
 */
int gpu_matmul(float *out, const float *A, const float *B, int R, int K, int C) {
    if (!gpu_ok) return -1;
    float *dA, *dB, *dO;
    size_t sa = (size_t)R * K * 4, sb = (size_t)K * C * 4, so = (size_t)R * C * 4;
    if (cudaMalloc(&dA, sa)) return -2;
    if (cudaMalloc(&dB, sb)) { cudaFree(dA); return -2; }
    if (cudaMalloc(&dO, so)) { cudaFree(dA); cudaFree(dB); return -2; }
    cudaMemcpy(dA, A, sa, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sb, cudaMemcpyHostToDevice);

    float alpha = 1.0f, beta = 0.0f;
    /* row-major: out[R,C] = A[R,K] @ B[K,C]
     * truco estandar: out_cm[C,R] = B_cm[C,K] @ A_cm[K,R]
     * donde B_cm = B^T (row-major B[K,C] == col-major B_cm[C,K] con ldb=C)
     * cublasSgemm(opA=N, opB=N, m=C, n=R, k=K, A=B (lda=C), B=A (ldb=K))
     * y resultado en col-major (C x R) = out row-major (R x C) exacto */
    cublasStatus_t st = cublasSgemm(hnd, CUBLAS_OP_N, CUBLAS_OP_N,
                                    C, R, K, &alpha,
                                    dB, C,   /* B col-major ldb=C */
                                    dA, K,   /* A col-major lda=K */
                                    &beta, dO, C);
    int ok = (st == CUBLAS_STATUS_SUCCESS);
    if (ok) cudaMemcpy(out, dO, so, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dO);
    return ok ? 0 : -3;
}

/* version batch para entrenamiento: copia W completa a VRAM una vez,
 * opera N matmuls, baja solo lo necesario (futuro: full GPU trainer) */
void gpu_warmup(void) {
    if (!gpu_ok) return;
    /* fuerza la creacion del contexto cublas */
    float a[4] = {1, 2, 3, 4}, b[4] = {1, 0, 0, 1}, o[4];
    gpu_matmul(o, a, b, 2, 2, 2);
}