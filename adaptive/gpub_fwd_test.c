/* gpub_fwd_test.c - FASE B verificado: logits GPU vs logits CPU (batch vs per-pos)
 * usa el ad_forward del core (KV cache) como referencia */
#include "adaptive.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int ad_gpub_init(void);
int ad_gpub_forward(AdModel *m, const unsigned char *bytes, int T, float *logits_out);

int main(int argc, char **argv) {
    const char *model = argc >= 2 ? argv[1] : "model_bpe2.adm";
    if (ad_gpub_init()) { printf("no GPU\n"); return 1; }

    AdModel m; memset(&m, 0, sizeof m);
    if (ad_load(&m, model)) { printf("no carga\n"); return 1; }
    printf("modelo vocab=%d dim=%d\n", m.cfg.vocab, m.cfg.dim);

    const char *txt = "Write";
    int T = (int)strlen(txt);
    if (T > m.cfg.max_seq - 2) T = m.cfg.max_seq - 2;

    /* GPU: forward batch */
    float *gpu_lg = (float *)malloc((size_t)T * AD_VOCAB * sizeof(float));
    ad_gpub_forward(&m, (const unsigned char *)txt, T, gpu_lg);

    /* CPU: forward pos a pos con el core */
    m.n_tok = 0;
    for (int t = 0; t < T; t++) m.tokens[m.n_tok++] = txt[t];
    m.n_prompt = m.n_tok;
    for (int i = 1; i <= m.n_tok; i++) {
        int save = m.n_tok;
        m.n_tok = i;
        ad_forward(&m);
        m.n_tok = save;
    }
    float err = 0.f;
    for (int v = 0; v < 256; v++)
        err += fabsf(gpu_lg[(size_t)(T-1) * AD_VOCAB + v] - m.logits[v]);
    err /= 256;
    printf("err medio logits (pos final): %.6f => %s\n",
           err, err < 0.5f ? "OK" : "FALLO");
    return err < 0.5f ? 0 : 1;
}
