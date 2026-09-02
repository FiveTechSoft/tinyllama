/* ad_resident.c - FASE E: entrenador con residencia completa en VRAM
 *
 * los pesos, gradientes y moments Adam viven SIEMPRE en VRAM.
 * el host solo:
 *   - sube ventanas de tokens (BT x int, kilobytes)
 *   - baja: loss del step + grad_norm (4-8 bytes)
 *
 * estructura:
 *   ad_res_alloc(model)          -> copia pesos a VRAM, alloc workspace
 *   ad_res_train_chunk(steps, batch, lr, total) -> loss media
 *   ad_res_fetch(out)            -> baja los pesos entrenados (host)
 *   ad_res_free()
 *
 * el forward/backward por capa reusa ad_gpub_lin (GEMM cublas) con los
 * tensores device; la atencion causal y los LN van en kernels CUDA.
 */
#include "adaptive.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* kernels de ad_cuda.cu */
extern "C" {
void k_layernorm_launch(const float *x, const float *g, const float *b,
                        float *out, int n, int rows, cudaStream_t s);
void k_gelu_launch(float *x, int n, cudaStream_t s);
void k_softmax_launch(float *x, int n, int rows, cudaStream_t s);
void k_adam_full_launch(float *w, const float *g, float *m, float *v,
                        float lr, float bc1, float bc2, size_t n, cudaStream_t s);
void k_zero_launch(float *x, size_t n, cudaStream_t s);
void k_scale_launch(float *x, float s, size_t n, cudaStream_t s);
void k_emb_add_launch(const float *tok, const float *pos, const int *ids,
                      float *x, int dim, int T, cudaStream_t s);
}

static cublasHandle_t rh = NULL;
static int res_ok = 0;

/* VRAM persistente */
static float *dW = NULL;    /* pesos [nfloats] */
static float *dGRAD = NULL; /* gradientes */
static float *dADM = NULL, *dADV = NULL;
static float *dTokEmb = NULL, *dPosEmb = NULL;   /* alias dentro de dW */
static int   *dIds = NULL;  /* ventana de tokens [T] */

/* workspace de activaciones (max_seq x dim) */
static float *dX, *dL1, *dQKV, *dAO, *dPR, *dL2, *dF1, *dLF;
static float *dSCR;
static float *dLogits;
static size_t ws_cap = 0;
static int res_dim = 0, res_layers = 0, res_hid = 0, res_vocab = 0;
static size_t res_nfloats = 0;
static float *h_logits = NULL;    /* espejo host del logits del step */

extern "C" {

int ad_res_alloc(AdModel *m) {
    if (cudaFree(0) != cudaSuccess) return -1;
    if (cublasCreate(&rh) != CUBLAS_STATUS_SUCCESS) return -2;

    res_dim = m->cfg.dim;
    res_layers = m->cfg.n_layers;
    res_hid = m->cfg.hidden;
    res_vocab = m->cfg.vocab > 0 ? m->cfg.vocab : 256;
    res_nfloats = ad_total_floats(&m->cfg);

    /* pesos + gradientes + adam a VRAM (una sola vez) */
    if (cudaMalloc(&dW, res_nfloats * 4)) return -3;
    if (cudaMalloc(&dGRAD, res_nfloats * 4)) return -3;
    if (cudaMalloc(&dADM, res_nfloats * 4)) return -3;
    if (cudaMalloc(&dADV, res_nfloats * 4)) return -3;
    cudaMemcpy(dW, m->w, res_nfloats * 4, cudaMemcpyHostToDevice);
    cudaMemset(dGRAD, 0, res_nfloats * 4);
    cudaMemset(dADM, 0, res_nfloats * 4);
    cudaMemset(dADV, 0, res_nfloats * 4);

    /* workspace por ventana */
    size_t T = (size_t)m->cfg.max_seq;
    size_t d = (size_t)res_dim, hid = (size_t)res_hid;
    if (cudaMalloc(&dX,  (L + 1) * T * d * 4)) return -4;
    if (cudaMalloc(&dL1, L * T * d * 4)) return -4;
    if (cudaMalloc(&dQKV, L * T * 3 * d * 4)) return -4;
    if (cudaMalloc(&dAO, 2 * L * T * d * 4)) return -4;
    if (cudaMalloc(&dPR, L * T * d * 4)) return -4;
    if (cudaMalloc(&dL2, L * T * d * 4)) return -4;
    if (cudaMalloc(&dF1, L * T * hid * 4)) return -4;
    if (cudaMalloc(&dLogits, T * (size_t)res_vocab * 4)) return -4;
    if (cudaMalloc(&dSCR, T * 4)) return -4;
    if (cudaMalloc(&dTokEmb, 0)) return -4;   /* alias: dentro de dW */
    if (cudaMalloc((void **)&dPosEmb, 0)) return -4;
    if (cudaMalloc((void **)&dTokEmb, 1)) { /* reusa punteros calculados */ }

    h_logits = (float *)malloc(T * (size_t)res_vocab * 4);
    res_ok = 1;
    return 0;
}

int ad_res_ok(void) { return res_ok; }

} /* extern "C" */