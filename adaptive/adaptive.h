/* adaptive.h - LLM adaptativo byte-level: nucleo comun
 *
 * Arquitectura: transformer decoder-only, vocabulario de 256 bytes
 * (0x01=<user 0x02=<bot 0x03=<eos> reservados como tokens de control)
 *
 * bloque(l):  x += bproj(Wproj @ attn(ln1(x)))
 *             x += b2(W2 @ gelu(W1 @ ln2(x)))
 * logits:     Whead @ lnf(x_final)
 *
 * Formato binario ADM (little-endian), 36 bytes de cabecera:
 *   'ADMF' | u32 ver=1 | u32 dim | u32 L | u32 H | u32 hidden | u32 max_seq
 *   | u32 train_steps | u32 flags
 *   seguido de los pesos F32 en el orden exacto de ad_layout.
 */
#ifndef ADAPTIVE_H
#define ADAPTIVE_H

#include <stdint.h>
#include <stddef.h>

#define AD_VOCAB   256
#define AD_MAX_SEQ 512
#define AD_MAX_LAYERS 12
#define AD_HDR     36
#define AD_VERSION 1u

/* bytes de control del vocabulario */
#define AD_CTL_USER 0x01 /* <u> inicio turno usuario */
#define AD_CTL_BOT  0x02 /* <b> inicio turno bot     */
#define AD_CTL_EOS  0x03 /* <e> fin de respuesta     */

typedef struct {
    int dim, n_layers, n_heads, hidden, max_seq;
    int head_dim;            /* dim / n_heads */
    uint32_t train_steps, flags;
} AdConfig;

typedef struct {
    size_t tok_emb, pos_emb, layers, per_layer;
    size_t lnf_g, lnf_b, w_head, b_head, total;
} AdLayout;

typedef struct {
    AdConfig cfg;
    AdLayout lay;
    float   *w;              /* arena de pesos [lay.total] */
    float   *logits;         /* [AD_VOCAB] */
    int     *tokens;         /* [max_seq] */
    int      n_tok, n_prompt;
    float   *x, *h, *qkv, *att, *proj, *f1, *hf;   /* scratch */
    float   *kc, *vc;        /* KV cache [L][max_seq][dim] */
} AdModel;

/* layout e IO */
void ad_layout_build(AdConfig *c, AdLayout *L);
size_t ad_total_floats(const AdConfig *c);
int  ad_model_alloc(AdModel *m);       /* usa m->cfg */
void ad_model_free(AdModel *m);
int  ad_init_fresh(AdModel *m, int dim, int n_layers, int n_heads, int seq);
int  ad_load(AdModel *m, const char *path);
int  ad_save(AdModel *m, const char *path);
int  ad_load_from(AdModel *m, const uint8_t *buf, int size);

/* inferencia */
void ad_forward(AdModel *m);                        /* 1 posicion + KV cache */
int  ad_set_prompt(AdModel *m, const char *text);   /* tokeniza + prefill */
int  ad_step(AdModel *m, float temp, int top_k);    /* -1 fin | byte 0..255 */
int  ad_generated_n(AdModel *m);                    /* tokens generados */
int  ad_byte(AdModel *m, int idx);                  /* token idx del stream */

/* kernels */
void  ad_matmul(float *out, const float *W, const float *x, int R, int C);
void  ad_matmul_T(float *out, const float *W, const float *x, int R, int C);
void  ad_layernorm(float *out, const float *x, const float *g, const float *b, int n);
void  ad_softmax(float *x, int n);
void  ad_gelu(float *x, int n);
void  ad_gelu_bwd(float *dx, const float *pre, int n);
float ad_randf(void);
float ad_gauss(void);

#endif