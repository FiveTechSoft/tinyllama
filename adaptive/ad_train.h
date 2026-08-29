/* ad_train.h - estructuras del entrenamiento (compartidas train.c/ad_train.c) */
#ifndef AD_TRAIN_H
#define AD_TRAIN_H

#include "adaptive.h"

/* activaciones y gradientes de UNA ventana (T posiciones) */
typedef struct {
    /* forward */
    float *x;        /* [L+1][T][dim] residual stream por capa */
    float *ln1_out;  /* [L][T][dim] */
    float *qkv;      /* [L][T][3dim] */
    float *ao;       /* [L][T][dim] heads concat (pre-Wproj) */
    float *proj;     /* [L][T][dim] salida Wproj */
    float *ln2_out;  /* [L][T][dim] */
    float *f1;       /* [L][T][hid] pre-gelu */
    float *f1a;      /* [L][T][hid] post-gelu */
    float *lnf_out;  /* [T][dim] tras LN final */
    float *logits;   /* [T][V] */
    /* backward */
    float *dx;       /* [T][dim] gradiente en el stream (scratch por capa) */
    float *dlogits;  /* [T][V] */
    float *dh;       /* [T*(dim>hid?dim:hid)] scratch */
    /* scratch atencion */
    float *scr;      /* [T] */
} WA;

extern WA  g_ws;
extern AdModel g_tm;          /* modelo de trabajo del trainer */

/* reserva buffers de la WA segun config del modelo m; 0 ok */
int  tr_alloc_work(AdModel *m);
void tr_free_work(void);

/* forward (+backward si do_bwd) de una ventana de T bytes.
 * bytes[0..T-1] con targets bytes[1..T] (needs T+1 bytes).
 * devuelve loss media por prediccion; gradientes acumulan en g_* si do_bwd */
float tr_window(AdModel *m, const uint8_t *bytes, int T, int do_bwd);

/* evalua PPL media sobre n ventanas aleatorias del corpus (len) */
float tr_eval_ppl(AdModel *m, const uint8_t *data, size_t len,
                  int T, int n_windows);

#endif
