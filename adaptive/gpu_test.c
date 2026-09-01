/* gpu_test.c - verifica cuBLAS: multiplica 2x2 y compara con CPU */
#include <stdio.h>

int gpu_init(void);
int gpu_available(void);
int gpu_matmul(float *out, const float *A, const float *B, int R, int K, int C);

int main(void) {
    int r = gpu_init();
    if (r) { printf("no GPU (err %d)\n", r); return 1; }
    printf("GPU OK: cuBLAS activo\n");
    float a[6] = {1, 2, 3, 4, 5, 6};   /* 2x3 */
    float b[6] = {7, 8, 9, 10, 11, 12}; /* 3x2 */
    float o[4];
    if (gpu_matmul(o, a, b, 2, 3, 2)) { printf("matmul fail\n"); return 1; }
    /* esperado: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58,64] fila1
     *           [4*7+5*9+6*11, 4*8+5*9+6*12] = [139,154] fila2 */
    printf("matmul 2x3 @ 3x2 = [%g %g %g %g]\n", o[0], o[1], o[2], o[3]);
    int ok = (o[0] == 58 && o[1] == 64 && o[2] == 139 && o[3] == 154);
    printf("%s\n", ok ? "cuBLAS matmul: OK" : "cuBLAS matmul: FALLO");
    return ok ? 0 : 1;
}