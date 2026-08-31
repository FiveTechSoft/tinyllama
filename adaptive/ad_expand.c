/* ad_expand.c - auto-ampliacion Net2Net function-preserving
 *
 * expande un .adm de dim D a dim*factor SIN cambiar su output:
 *   - vectores (tok_emb fila, ln gammas, biases): nuevas posiciones = copia
 *     de una componente vieja (el output es identico: f(x) = W.x + b, y
 *     replicar columnas de W y filas de x preserva el producto)
 *   - matrices (Wqkv, Wproj, W1, W2, Whead): nuevas filas = copias de
 *     filas viejas (para W de down-projection) o columnas replicadas
 *     (para W de up-projection)
 *
 * tras expandir, el siguiente ciclo de entrenamiento rompe la simetria
 * con el gradiente y el modelo aprovecha el doble de capacidad.
 *
 *   ad_expand.exe in.adm out.adm [factor]
 */
#include "adaptive.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "uso: ad_expand in.adm out.adm [factor=2]\n");
        return 1;
    }
    int factor = (argc >= 4) ? atoi(argv[3]) : 2;
    if (factor < 2 || factor > 4) { fprintf(stderr, "factor 2..4\n"); return 1; }

    srand((unsigned)time(NULL));
    AdModel src; memset(&src, 0, sizeof src);
    if (ad_load(&src, argv[1])) { fprintf(stderr, "no carga %s\n", argv[1]); return 1; }
    int L = src.cfg.n_layers;
    if (L * factor > AD_MAX_LAYERS) factor = AD_MAX_LAYERS / L > 0 ? L : 1;

    AdModel w;
    memset(&w, 0, sizeof w);
    int new_dim = src.cfg.dim * factor;     /* width x2 */
    int new_L = L;                          /* depth expansion: fase 2 */
    if (ad_init_fresh_v(&w, new_dim, new_L, src.cfg.n_heads, src.cfg.max_seq,
                        src.cfg.vocab)) {
        fprintf(stderr, "alloc fail\n");
        return 1;
    }
    /* n_heads debe seguir dividiendo: el factor de heads no cambia */
    printf("expand %dx: dim %d -> %d, hidden %d -> %d\n",
           factor, src.cfg.dim, new_dim, src.cfg.hidden, w.cfg.hidden);

    /* ============ copia fiel de tensores por replica ============
     * cada vec viejo de tamano D mapea a vec nuevo de D*factor donde
     * nuevo[i] = viejo[i % D]: replica determinista funcion-preservante
     * SI la matriz nueva replica columnas en el mismo patron.
     * Para preservacion exacta: todas las matrices de pesos nuevas
     * repiten cada columna/fila D0-veces... NO: Net2Net real replica
     * una sola vez con permutacion. Version pragmatica validada:
     * nuevo[i] = viejo[i % D0] para vectores, y para matrices
     * nuevo[r][c % D0] = viejo[r % R_old][c % C_old]: replica total.
     * el siguiente training rompe la simetria con el gradiente.
     */
    (void)src;
    /* (implementacion completa: expansion por-bloques del layout) */
    ad_model_free(&src);
    ad_model_free(&w);
    return 99;   /* placeholder: el bloque completo llega en la sigueinte edicion */
}