/* ad_gpu.h - GPU-accelerated training for the adaptive model
 * Uses cuBLAS for batched GEMM (all T positions at once per layer)
 * and CUDA kernels for LN, GELU, attention, Adam.
 * Falls back to CPU if no GPU available. */
#ifndef AD_GPU_H
#define AD_GPU_H

#include "adaptive.h"

#ifdef __cplusplus
extern "C" {
#endif

int   ad_gpu_init(AdModel *m);
float ad_gpu_fwd_bwd(AdModel *m, const int *tokens, int T, int do_bwd);
void  ad_gpu_adam_step(float lr, float beta1, float beta2, float eps, float wd);
void  ad_gpu_download(AdModel *m);
void  ad_gpu_free(void);
int   ad_gpu_ready(void);

#ifdef __cplusplus
}
#endif

#endif
