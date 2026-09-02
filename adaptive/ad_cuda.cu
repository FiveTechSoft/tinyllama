/* ad_cuda.cu - kernels CUDA del entrenador residente (FASE E)
 *
 * todo vive en VRAM: pesos, gradientes, moments Adam. El host solo envia
 * ventanas de tokens y recibe loss. Patron Jalapeno: residencia completa,
 * transfers minimos.
 *
 * kernels:
 *   k_layernorm  : LN por fila
 *   k_gelu       : elementwise
 *   k_adam       : elementwise (update con gradiente global escalado)
 *   k_softmax_row: softmax por fila (scores de atencion)
 *   k_emb_lookup : X[t] = tok_emb[id] + pos_emb[t]
 *
 * las GEMM van via cublasSgemm (ad_gpub.c ya verificado)
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

/* ---------------- elementwise kernels ---------------- */

__global__ void k_layernorm(const float *x, const float *g, const float *b,
                            float *out, int n) {
    int row = blockIdx.x;
    int i = threadIdx.x;
    extern __shared__ float sm[];
    /* media y var en shared (n <= 1024 threads) */
    float v = (i < n) ? x[(size_t)row * n + i] : 0.f;
    sm[i] = v;
    __syncthreads();
    if (i == 0) {
        float m = 0.f, var = 0.f;
        for (int k = 0; k < n; k++) m += sm[k];
        m /= n;
        for (int k = 0; k < n; k++) { float d = sm[k] - m; var += d * d; }
        var /= n;
        sm[n] = m;            /* guardamos m en sm[n] */
        sm[n + 1] = 1.0f / sqrtf(var + 1e-5f);   /* s en sm[n+1] */
    }
    __syncthreads();
    float m = sm[n], s = sm[n + 1];
    out[(size_t)row * n + i] = (v - m) * s * g[i] + b[i];
}

__global__ void k_gelu(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

__global__ void k_softmax(float *x, int n) {
    int row = blockIdx.x;
    /* softmax de fila n elementos: un thread por elemento con reduccion
     * simple (n <= 256: un bloque es suficiente) */
    extern __shared__ float sm[];
    int i = threadIdx.x;
    float v = (i < n) ? x[(size_t)row * n + i] : -1e30f;
    sm[i] = v;
    __syncthreads();
    if (i == 0) {
        float mx = sm[0];
        for (int k = 1; k < n; k++) if (sm[k] > mx) mx = sm[k];
        float s = 0.f;
        for (int k = 0; k < n; k++) { sm[k] = expf(sm[k] - mx); s += sm[k]; }
        sm[n] = s;
    }
    __syncthreads();
    if (i < n) x[(size_t)row * n + i] = sm[i] / sm[n];
}

/* Adam update elementwise: w -= lr * mhat / (sqrt(vhat) + eps)
 * gradiente ya escalado por batch y clip en dGRAD */
/* adam con gradientes en VRAM: dGRAD ya contiene el gradiente medio */
__global__ void k_adam_full(float *w, const float *g, float *m, float *v,
                            float lr, float bc1, float bc2, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i];
    m[i] = 0.9f * m[i] + 0.1f * gi;
    v[i] = 0.999f * v[i] + 0.001f * gi * gi;
    w[i] -= lr * (m[i] / bc1) / (sqrtf(v[i] / bc2) + 1e-8f);
}

__global__ void k_zero(float *x, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.f;
}

__global__ void k_scale(float *x, float s, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

__global__ void k_emb_add(const float *tok_emb, const float *pos_emb,
                          const int *ids, float *x, int dim, int T) {
    int t = blockIdx.x;
    int i = threadIdx.x;
    if (i < dim) {
        int id = ids[t];
        x[(size_t)t * dim + i] = tok_emb[(size_t)id * dim + i]
                               + pos_emb[(size_t)t * dim + i];
    }
}
