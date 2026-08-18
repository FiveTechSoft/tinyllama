/*
 * tinyllama.c - Motor de inferencia C puro para TinyLlama-1.1B (GGUF Q4_0)
 *
 * Basado en llm_inference.c de fivetechsoft/dreaming (motor C puro, cero deps).
 * Adaptado para:
 *   - Pesos Q4_0 des cuantizados on-the-fly (el modelo real es Q4_0/Q6_K,
 *     no F16) -> la memoria se mantiene en ~1 GB y cabe en WebAssembly.
 *   - output.weight (Q6_K) se descuantiza a F32 una sola vez al cargar.
 *   - Sin OpenMP (WASM single-thread).
 *   - Exports para Emscripten: tl_init / tl_set_prompt / tl_step / tl_token_str.
 *
 * Nativo:
 *   gcc -O2 -o tinyllama tinyllama.c -lm
 *   ./tinyllama model.gguf "The secret to happiness is" 30 0.8 25
 *
 * WASM (GitHub Action, emscripten):
 *   emcc -O3 -msimd128 -s ALLOW_MEMORY_GROWTH=1 -s EXPORTED_RUNTIME_METHODS=ccall ...
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define TL_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define TL_EXPORT
#endif

/* ================================================================
 *  CONSTANTS (TinyLlama-1.1B)
 * ================================================================ */
#define HIDDEN_DIM    2048
#define NUM_LAYERS    22
#define NUM_HEADS     32    /* Q heads */
#define NUM_KV_HEADS  4     /* KV heads (GQA: 32/4 = 8 Q per KV) */
#define HEAD_DIM      64
#define KV_DIM        (NUM_KV_HEADS * HEAD_DIM)  /* 256 */
#define VOCAB_SIZE    32000
#define FFN_DIM       5632
#define MAX_SEQ       2048
#define MAX_TOKENS    1024

#define GGUF_F32   0
#define GGUF_F16   1
#define GGUF_Q4_0  2
#define GGUF_Q6_K 14

#define QK4_0 32
#define QK_K  256

/* ================================================================
 *  HALF-PRECISION FLOAT (F16 -> F32)
 * ================================================================ */
static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = (uint32_t)h & 0x3FF;
    uint32_t bits;

    if (exp == 0) {
        /* subnormal (o cero): valor = man * 2^-24, exacto en f32 */
        float f = (float)man * 5.9604644775390625e-08f;
        memcpy(&bits, &f, 4);
        bits |= sign;
        memcpy(&f, &bits, 4);
        return f;
    }
    if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static inline float read_f16(const uint8_t *p) {
    uint16_t h;
    memcpy(&h, p, 2);
    return f16_to_f32(h);
}

/* ================================================================
 *  QUANTIZED DOT PRODUCTS (weights stay quantized in memory)
 * ================================================================ */

/* Q4_0 block: fp16 scale d + 16 bytes of nibbles (32 values).
 * y[i]    = (qs[i] & 0x0F) - 8   for i in [0,16)
 * y[i+16] = (qs[i] >> 4)   - 8   for i in [0,16)
 * value = nibble * d                                        */
static float dot_q4_0(const float *x, const uint8_t *w, int n) {
    float sum = 0.0f;
    int nb = n / QK4_0;
    for (int b = 0; b < nb; b++) {
        float d = read_f16(w);
        const uint8_t *qs = w + 2;
        const float *xb = x + b * QK4_0;
        float bs = 0.0f;
        for (int i = 0; i < 16; i++) {
            int lo = (qs[i] & 0x0F) - 8;
            int hi = (qs[i] >> 4)   - 8;
            bs += lo * xb[i] + hi * xb[i + 16];
        }
        sum += d * bs;
        w += 18; /* 2 + 16 bytes per block */
    }
    return sum;
}

/* Q6_K super-block (256 values): ql[128] + qh[64] + scales[16] + d(fp16) */
static void dequant_q6_K(const uint8_t *w, float *y, int64_t n) {
    int64_t nb = n / QK_K;
    for (int64_t b = 0; b < nb; b++) {
        const uint8_t *ql = w;
        const uint8_t *qh = w + 128;
        const int8_t  *sc = (const int8_t *)(w + 192);
        float d = read_f16(w + 208);

        for (int nn = 0; nn < 2; nn++) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16; /* solo indexa scales; los shifts de qh son fijos */
                int8_t q1 = (int8_t)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l] >> 4)       | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l]      = d * sc[is]     * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            ql += 64; qh += 32; sc += 8; y += 128;
        }
        w += 210;
    }
}

/* ================================================================
 *  GGUF READER (from memory buffer)
 * ================================================================ */
typedef struct {
    char name[96];
    const uint8_t *ptr;   /* points into the model buffer */
    int type;             /* GGUF_F32 / GGUF_F16 / GGUF_Q4_0 / GGUF_Q6_K */
    int64_t rows;         /* out_dim (dims[1] if 2D, else 1) */
    int64_t cols;         /* in_dim  (dims[0]) */
} Tensor;

typedef struct {
    uint8_t *data;        /* we do NOT own this pointer */
    int64_t size;
    Tensor tensors[256];
    int num_tensors;
    char meta_keys[64][64];
    int64_t meta_ints[64];
    int num_meta_int;
} GGUF;

static uint64_t rd_u64(const uint8_t *d, int64_t *p) {
    uint64_t v; memcpy(&v, d + *p, 8); *p += 8; return v;
}
static uint32_t rd_u32(const uint8_t *d, int64_t *p) {
    uint32_t v; memcpy(&v, d + *p, 4); *p += 4; return v;
}
static void rd_str(const uint8_t *d, int64_t *p, char *out, int maxlen) {
    uint64_t len = rd_u64(d, p);
    uint64_t c = len < (uint64_t)(maxlen - 1) ? len : (uint64_t)(maxlen - 1);
    memcpy(out, d + *p, c); out[c] = '\0';
    *p += len;
}

static const int gguf_tsz[13] = {1,1,2,2,4,4,4,1,0,0,8,8,8};

/* tokenizer vocab (filled during metadata parse) */
static char *g_vocab[VOCAB_SIZE + 8];
static int g_vocab_size = 0;

static void gguf_parse(GGUF *g, uint8_t *buf, int64_t size) {
    g->data = buf; g->size = size;
    g->num_tensors = 0; g->num_meta_int = 0;

    if (size < 24 || memcmp(buf, "GGUF", 4) != 0) {
        fprintf(stderr, "Bad GGUF magic\n"); exit(1);
    }
    int64_t p = 4;
    uint32_t version = rd_u32(buf, &p);
    int64_t tcount = (int64_t)rd_u64(buf, &p);
    int64_t kvcount = (int64_t)rd_u64(buf, &p);
    if (version != 3) fprintf(stderr, "Warning: GGUF version %u\n", version);

    for (int64_t i = 0; i < kvcount; i++) {
        char key[96];
        rd_str(buf, &p, key, 96);
        uint32_t vtype = rd_u32(buf, &p);

        if (vtype == 8) { /* STRING: skip */
            uint64_t sl = rd_u64(buf, &p);
            p += (int64_t)sl;
        } else if (vtype == 9) { /* ARRAY */
            uint32_t atype = rd_u32(buf, &p);
            uint64_t alen = rd_u64(buf, &p);
            if (strcmp(key, "tokenizer.ggml.tokens") == 0 && atype == 8) {
                g_vocab_size = (int)alen;
                if (g_vocab_size > VOCAB_SIZE + 8) g_vocab_size = VOCAB_SIZE + 8;
                for (int a = 0; a < g_vocab_size; a++) {
                    uint64_t sl = rd_u64(buf, &p);
                    g_vocab[a] = malloc(sl + 1);
                    memcpy(g_vocab[a], buf + p, sl);
                    g_vocab[a][sl] = '\0';
                    p += (int64_t)sl;
                }
                /* skip any remaining entries */
                for (uint64_t a = (uint64_t)g_vocab_size; a < alen; a++) {
                    uint64_t sl = rd_u64(buf, &p);
                    p += (int64_t)sl;
                }
            } else {
                for (uint64_t a = 0; a < alen; a++) {
                    if (atype == 8) { uint64_t sl = rd_u64(buf, &p); p += (int64_t)sl; }
                    else p += gguf_tsz[atype < 13 ? atype : 0];
                }
            }
        } else { /* scalar */
            int64_t v = 0;
            switch (vtype) {
                case 0: { uint8_t  x; memcpy(&x, buf + p, 1); v = x; } break;
                case 1: { int8_t   x; memcpy(&x, buf + p, 1); v = x; } break;
                case 2: { uint16_t x; memcpy(&x, buf + p, 2); v = x; } break;
                case 3: { int16_t  x; memcpy(&x, buf + p, 2); v = x; } break;
                case 4: { uint32_t x; memcpy(&x, buf + p, 4); v = x; } break;
                case 5: { int32_t  x; memcpy(&x, buf + p, 4); v = x; } break;
                case 7: { uint8_t  x; memcpy(&x, buf + p, 1); v = x; } break;
                case 10: { uint64_t x; memcpy(&x, buf + p, 8); v = (int64_t)x; } break;
                case 11: { int64_t  x; memcpy(&x, buf + p, 8); v = x; } break;
                default: break; /* F32/F64: skip */
            }
            p += gguf_tsz[vtype < 13 ? vtype : 0];
            if (g->num_meta_int < 64) {
                strncpy(g->meta_keys[g->num_meta_int], key, 63);
                g->meta_keys[g->num_meta_int][63] = '\0';
                g->meta_ints[g->num_meta_int] = v;
                g->num_meta_int++;
            }
        }
    }

    /* tensor index */
    int64_t dims[8];
    int64_t raw_off[256];
    int ndims_arr[256];
    for (int64_t i = 0; i < tcount && g->num_tensors < 256; i++) {
        Tensor *t = &g->tensors[g->num_tensors];
        rd_str(buf, &p, t->name, 96);
        uint32_t ndims = rd_u32(buf, &p);
        int64_t nelem = 1;
        for (uint32_t d = 0; d < ndims && d < 8; d++) {
            dims[d] = (int64_t)rd_u64(buf, &p);
            nelem *= dims[d];
        }
        t->type = (int)rd_u32(buf, &p);
        raw_off[g->num_tensors] = (int64_t)rd_u64(buf, &p);
        ndims_arr[g->num_tensors] = ndims;
        t->cols = ndims >= 1 ? dims[0] : 1;
        t->rows = ndims >= 2 ? dims[1] : 1;
        (void)nelem;
        g->num_tensors++;
    }

    /* data section: aligned to 32 bytes after the index */
    int64_t data_start = ((p + 31) / 32) * 32;
    for (int i = 0; i < g->num_tensors; i++) {
        g->tensors[i].ptr = buf + data_start + raw_off[i];
    }
}

static int64_t gguf_meta_int(GGUF *g, const char *key, int64_t def) {
    for (int i = 0; i < g->num_meta_int; i++)
        if (strcmp(g->meta_keys[i], key) == 0) return g->meta_ints[i];
    return def;
}

static Tensor *gguf_find(GGUF *g, const char *name) {
    for (int i = 0; i < g->num_tensors; i++)
        if (strcmp(g->tensors[i].name, name) == 0) return &g->tensors[i];
    fprintf(stderr, "tensor '%s' not found\n", name);
    exit(1);
}

static int64_t tensor_row_bytes(const Tensor *t) {
    switch (t->type) {
        case GGUF_F32:  return t->cols * 4;
        case GGUF_F16:  return t->cols * 2;
        case GGUF_Q4_0: return (t->cols / QK4_0) * 18;
        case GGUF_Q6_K: return (t->cols / QK_K) * 210;
        default: fprintf(stderr, "unsupported tensor type %d\n", t->type); exit(1);
    }
}

/* ================================================================
 *  TOKENIZER (BPE, vocab from GGUF)
 * ================================================================ */

static int vocab_find(const char *piece) {
    for (int v = 0; v < g_vocab_size; v++)
        if (g_vocab[v] && strcmp(g_vocab[v], piece) == 0) return v;
    return -1;
}

/* Encode text into token ids. Llama BPE: spaces become U+2581 (e2 96 81),
 * and a leading U+2581 is prepended (sentence start). Initial symbols are
 * UTF-8 characters mapped to vocab tokens (byte fallback <0xXX> = id b+3
 * when the char is not in vocab). Merge rank is approximated by vocab
 * index (true for Llama vocabularies). */
static int tokenize(const char *text, int *tokens, int max_tokens) {
    /* build piece buffer with space -> \xE2\x96\x81 and leading marker */
    static char buf[8192];
    int bn = 0;
    buf[bn++] = (char)0xE2; buf[bn++] = (char)0x96; buf[bn++] = (char)0x81;
    for (int i = 0; text[i] && bn < 8100; i++) {
        if (text[i] == ' ') {
            buf[bn++] = (char)0xE2; buf[bn++] = (char)0x96; buf[bn++] = (char)0x81;
        } else {
            buf[bn++] = text[i];
        }
    }
    buf[bn] = '\0';

    int ids[4096];
    static char pieces[4096][96];
    int n = 0;

    /* initial: one symbol per UTF-8 character */
    for (int i = 0; buf[i] && n < 4090;) {
        unsigned char b = (unsigned char)buf[i];
        int clen = 1;
        if ((b & 0xF0) == 0xF0) clen = 4;
        else if ((b & 0xE0) == 0xE0) clen = 3;
        else if ((b & 0xC0) == 0xC0) clen = 2;
        /* clamp to remaining bytes */
        int rem = bn - i;
        if (clen > rem) clen = 1;

        char piece[8] = {0};
        memcpy(piece, buf + i, clen);
        int id = clen > 1 ? vocab_find(piece) : -1;
        if (clen == 1) id = vocab_find(piece); /* single chars incl. ASCII */
        if (id >= 0) {
            ids[n] = id;
            memcpy(pieces[n], piece, clen);
            pieces[n][clen] = '\0';
            n++;
            i += clen;
        } else {
            /* byte fallback: <0xXX> token ids 3..258, piece = raw byte */
            for (int k = 0; k < clen && n < 4090; k++) {
                unsigned char bb = (unsigned char)buf[i + k];
                ids[n] = bb + 3;
                pieces[n][0] = (char)bb;
                pieces[n][1] = '\0';
                n++;
            }
            i += clen;
        }
    }

    /* BPE merges: repeatedly merge the pair whose merged token has the
     * lowest vocab index (approximation of merge rank) */
    for (;;) {
        int best_id = -1, best_pos = -1;
        int best_rank = g_vocab_size + 1;
        for (int i = 0; i < n - 1; i++) {
            char merged[192];
            snprintf(merged, sizeof(merged), "%s%s", pieces[i], pieces[i + 1]);
            int v = vocab_find(merged);
            if (v >= 0 && v < best_rank) {
                best_rank = v; best_id = v; best_pos = i;
            }
        }
        if (best_pos < 0) break;
        ids[best_pos] = best_id;
        strncpy(pieces[best_pos], g_vocab[best_id], 95);
        pieces[best_pos][95] = '\0';
        for (int i = best_pos + 1; i < n - 1; i++) {
            ids[i] = ids[i + 1];
            strcpy(pieces[i], pieces[i + 1]);
        }
        n--;
    }

    int out = n < max_tokens ? n : max_tokens;
    memcpy(tokens, ids, out * sizeof(int));
    return out;
}

/* Decode one token id into a static string, U+2581 -> ' ',
 * byte-fallback "<0xHH>" -> byte crudo */
static const char *decode_token(int id) {
    static char out[256];
    out[0] = '\0';
    if (id < 0 || id >= g_vocab_size || !g_vocab[id]) return out;
    const char *s = g_vocab[id];
    if (s[0] == '<' && s[1] == '0' && s[2] == 'x'
        && s[3] && s[4] && s[5] == '>' && s[6] == '\0') {
        int v = 0;
        for (int i = 3; i < 5; i++) {
            char c = s[i];
            v = v * 16 + (c <= '9' ? c - '0' : (c | 32) - 'a' + 10);
        }
        out[0] = (char)v;
        out[1] = '\0';
        return out;
    }
    int j = 0;
    for (int i = 0; s[i] && j < 250; i++) {
        if ((unsigned char)s[i] == 0xE2 && (unsigned char)s[i+1] == 0x96
            && (unsigned char)s[i+2] == 0x81) {
            out[j++] = ' ';
            i += 2;
        } else {
            out[j++] = s[i];
        }
    }
    out[j] = '\0';
    return out;
}

/* ================================================================
 *  MODEL
 * ================================================================ */
typedef struct {
    GGUF gguf;

    Tensor *w_emb;          /* Q4_0 [HIDDEN x VOCAB] */
    float  *w_output_f32;   /* dequantized output.weight (Q6_K -> F32) */
    Tensor *w_output_norm;

    Tensor *attn_norm[NUM_LAYERS];
    Tensor *attn_q[NUM_LAYERS];
    Tensor *attn_k[NUM_LAYERS];
    Tensor *attn_v[NUM_LAYERS];
    Tensor *attn_out[NUM_LAYERS];
    Tensor *ffn_norm[NUM_LAYERS];
    Tensor *ffn_gate[NUM_LAYERS];
    Tensor *ffn_up[NUM_LAYERS];
    Tensor *ffn_down[NUM_LAYERS];

    /* KV cache: [NUM_LAYERS][MAX_SEQ][KV_DIM] */
    float *k_cache[NUM_LAYERS];
    float *v_cache[NUM_LAYERS];
    int seq_len;

    /* generation state */
    int tokens[MAX_TOKENS];
    int n_tokens;      /* prompt + generated so far */
    int n_prompt;
    float logits[VOCAB_SIZE];

    /* temp buffers */
    float *tmp_h, *tmp_residual, *tmp_Q, *tmp_K, *tmp_V;
    float *tmp_attn, *tmp_gate, *tmp_up, *tmp_ffn, *tmp_ffn_out;
    float *emb_row;
} Model;

static Model g_model;

static void model_load(Model *m, uint8_t *buf, int64_t size) {
    gguf_parse(&m->gguf, buf, size);

    int hidden = (int)gguf_meta_int(&m->gguf, "llama.embedding_length", HIDDEN_DIM);
    int layers = (int)gguf_meta_int(&m->gguf, "llama.block_count", NUM_LAYERS);
    int heads  = (int)gguf_meta_int(&m->gguf, "llama.attention.head_count", NUM_HEADS);
    int kvh    = (int)gguf_meta_int(&m->gguf, "llama.attention.head_count_kv", NUM_KV_HEADS);
    int ffn    = (int)gguf_meta_int(&m->gguf, "llama.feed_forward_length", FFN_DIM);
    printf("Config: hidden=%d layers=%d heads=%d kv_heads=%d ffn=%d vocab=%d\n",
           hidden, layers, heads, kvh, ffn, g_vocab_size);
    if (hidden != HIDDEN_DIM || layers != NUM_LAYERS || heads != NUM_HEADS
        || kvh != NUM_KV_HEADS || ffn != FFN_DIM) {
        fprintf(stderr, "ERROR: model is not TinyLlama-1.1B architecture\n");
        exit(1);
    }

    m->w_emb = gguf_find(&m->gguf, "token_embd.weight");
    m->w_output_norm = gguf_find(&m->gguf, "output_norm.weight");

    /* output.weight is Q6_K in this file: dequantize once to F32 */
    Tensor *wout = gguf_find(&m->gguf, "output.weight");
    m->w_output_f32 = malloc((size_t)wout->rows * wout->cols * sizeof(float));
    if (!m->w_output_f32) { fprintf(stderr, "malloc output failed\n"); exit(1); }
    if (wout->type == GGUF_Q6_K) {
        int64_t rb = tensor_row_bytes(wout);
        for (int64_t r = 0; r < wout->rows; r++)
            dequant_q6_K(wout->ptr + r * rb, m->w_output_f32 + r * wout->cols, wout->cols);
    } else if (wout->type == GGUF_F32) {
        memcpy(m->w_output_f32, wout->ptr, (size_t)wout->rows * wout->cols * 4);
    } else {
        fprintf(stderr, "output.weight type %d unsupported\n", wout->type);
        exit(1);
    }

    for (int l = 0; l < NUM_LAYERS; l++) {
        char name[96];
        snprintf(name, 96, "blk.%d.attn_norm.weight", l);
        m->attn_norm[l] = gguf_find(&m->gguf, name);
        snprintf(name, 96, "blk.%d.attn_q.weight", l);
        m->attn_q[l] = gguf_find(&m->gguf, name);
        snprintf(name, 96, "blk.%d.attn_k.weight", l);
        m->attn_k[l] = gguf_find(&m->gguf, name);
        snprintf(name, 96, "blk.%d.attn_v.weight", l);
        m->attn_v[l] = gguf_find(&m->gguf, name);
        snprintf(name, 96, "blk.%d.attn_output.weight", l);
        m->attn_out[l] = gguf_find(&m->gguf, name);
        snprintf(name, 96, "blk.%d.ffn_norm.weight", l);
        m->ffn_norm[l] = gguf_find(&m->gguf, name);
        snprintf(name, 96, "blk.%d.ffn_gate.weight", l);
        m->ffn_gate[l] = gguf_find(&m->gguf, name);
        snprintf(name, 96, "blk.%d.ffn_up.weight", l);
        m->ffn_up[l] = gguf_find(&m->gguf, name);
        snprintf(name, 96, "blk.%d.ffn_down.weight", l);
        m->ffn_down[l] = gguf_find(&m->gguf, name);
    }

    for (int l = 0; l < NUM_LAYERS; l++) {
        m->k_cache[l] = calloc(MAX_SEQ * KV_DIM, sizeof(float));
        m->v_cache[l] = calloc(MAX_SEQ * KV_DIM, sizeof(float));
    }
    m->seq_len = 0;
    m->n_tokens = 0;
    m->n_prompt = 0;

    m->tmp_h        = malloc(HIDDEN_DIM * sizeof(float));
    m->tmp_residual = malloc(HIDDEN_DIM * sizeof(float));
    m->tmp_Q        = malloc(HIDDEN_DIM * sizeof(float));
    m->tmp_K        = malloc(KV_DIM * sizeof(float));
    m->tmp_V        = malloc(KV_DIM * sizeof(float));
    m->tmp_attn     = malloc(HIDDEN_DIM * sizeof(float));
    m->tmp_gate     = malloc(FFN_DIM * sizeof(float));
    m->tmp_up       = malloc(FFN_DIM * sizeof(float));
    m->tmp_ffn      = malloc(FFN_DIM * sizeof(float));
    m->tmp_ffn_out  = malloc(HIDDEN_DIM * sizeof(float));
    m->emb_row      = malloc(HIDDEN_DIM * sizeof(float));
}

/* ================================================================
 *  TRANSFORMER OPERATIONS
 * ================================================================ */
static void rmsnorm(float *out, const float *x, const float *w, int dim, float eps) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) sum += x[i] * x[i];
    float scale = 1.0f / sqrtf(sum / dim + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * scale * w[i];
}

static inline float dot_f32(const float *a, const float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

static inline float dot_f16(const float *a, const uint8_t *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * read_f16(b + i * 2);
    return sum;
}

/* out[j] = x . W_row(j); dispatch on the tensor's quantization type */
static void matmul_t(float *out, const float *x, const Tensor *W, int in_dim, int out_dim) {
    int64_t rb = tensor_row_bytes(W);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < out_dim; j++) {
        const uint8_t *row = W->ptr + j * rb;
        switch (W->type) {
            case GGUF_F32:  out[j] = dot_f32(x, (const float *)row, in_dim); break;
            case GGUF_F16:  out[j] = dot_f16(x, row, in_dim); break;
            case GGUF_Q4_0: out[j] = dot_q4_0(x, row, in_dim); break;
            default: fprintf(stderr, "matmul: unsupported type %d\n", W->type); exit(1);
        }
    }
}

static void matmul_f32(float *out, const float *x, const float *W, int in_dim, int out_dim) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < out_dim; j++)
        out[j] = dot_f32(x, W + (int64_t)j * in_dim, in_dim);
}

/* dequantize one embedding row (Q4_0 or F16/F32) */
static void embed_row(Model *m, int tok, float *out) {
    const Tensor *t = m->w_emb;
    const uint8_t *row = t->ptr + (int64_t)tok * tensor_row_bytes(t);
    if (t->type == GGUF_Q4_0) {
        int nb = HIDDEN_DIM / QK4_0;
        for (int b = 0; b < nb; b++) {
            float d = read_f16(row);
            const uint8_t *qs = row + 2;
            for (int i = 0; i < 16; i++) {
                out[b * 32 + i]      = ((qs[i] & 0x0F) - 8) * d;
                out[b * 32 + i + 16] = ((qs[i] >> 4)   - 8) * d;
            }
            row += 18;
        }
    } else if (t->type == GGUF_F16) {
        for (int i = 0; i < HIDDEN_DIM; i++) out[i] = read_f16(row + i * 2);
    } else {
        memcpy(out, row, HIDDEN_DIM * sizeof(float));
    }
}

/* RoPE tables */
static float rope_cos[MAX_SEQ][HEAD_DIM];
static float rope_sin[MAX_SEQ][HEAD_DIM];
static int rope_ready = 0;

static void rope_init(void) {
    for (int pos = 0; pos < MAX_SEQ; pos++) {
        for (int i = 0; i < HEAD_DIM; i += 2) {
            float freq = 1.0f / powf(10000.0f, (float)i / HEAD_DIM);
            float angle = pos * freq;
            rope_cos[pos][i] = rope_cos[pos][i+1] = cosf(angle);
            rope_sin[pos][i] = rope_sin[pos][i+1] = sinf(angle);
        }
    }
    rope_ready = 1;
}

static void apply_rope(float *x, int pos) {
    for (int i = 0; i < HEAD_DIM; i += 2) {
        float c = rope_cos[pos][i];
        float s = rope_sin[pos][i];
        float x0 = x[i], x1 = x[i + 1];
        x[i]     = x0 * c - x1 * s;
        x[i + 1] = x0 * s + x1 * c;
    }
}

/* forward pass for the token at position pos = m->n_tokens - 1 */
static int tl_trace = 0;
static void model_forward(Model *m) {
    if (!rope_ready) rope_init();
    int cur = m->n_tokens - 1;
    int tok = m->tokens[cur];
    if (tok < 0 || tok >= g_vocab_size) tok = 0;

    float *h = m->tmp_h;
    float *residual = m->tmp_residual;
    float *Q = m->tmp_Q, *K = m->tmp_K, *V = m->tmp_V;

    embed_row(m, tok, h);

    for (int l = 0; l < NUM_LAYERS; l++) {
        memcpy(residual, h, HIDDEN_DIM * sizeof(float));

        rmsnorm(h, h, (const float *)m->attn_norm[l]->ptr, HIDDEN_DIM, 1e-6f);

        matmul_t(Q, h, m->attn_q[l], HIDDEN_DIM, HIDDEN_DIM);
        matmul_t(K, h, m->attn_k[l], HIDDEN_DIM, KV_DIM);
        matmul_t(V, h, m->attn_v[l], HIDDEN_DIM, KV_DIM);

        for (int hd = 0; hd < NUM_HEADS; hd++)    apply_rope(Q + hd * HEAD_DIM, cur);
        for (int hd = 0; hd < NUM_KV_HEADS; hd++) apply_rope(K + hd * HEAD_DIM, cur);

        memcpy(m->k_cache[l] + cur * KV_DIM, K, KV_DIM * sizeof(float));
        memcpy(m->v_cache[l] + cur * KV_DIM, V, KV_DIM * sizeof(float));

        float *attn = m->tmp_attn;
        memset(attn, 0, HIDDEN_DIM * sizeof(float));
        int group = NUM_HEADS / NUM_KV_HEADS; /* 8 */

        for (int qh = 0; qh < NUM_HEADS; qh++) {
            int kvh = qh / group;
            float *q_h = Q + qh * HEAD_DIM;

            float scores[MAX_SEQ];
            float maxv = -1e30f;
            for (int k = 0; k <= cur; k++) {
                float *k_h = m->k_cache[l] + k * KV_DIM + kvh * HEAD_DIM;
                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) dot += q_h[d] * k_h[d];
                scores[k] = dot / sqrtf((float)HEAD_DIM);
                if (scores[k] > maxv) maxv = scores[k];
            }
            float sum = 0.0f;
            for (int k = 0; k <= cur; k++) {
                scores[k] = expf(scores[k] - maxv);
                sum += scores[k];
            }
            float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int k = 0; k <= cur; k++) {
                float wt = scores[k] * inv;
                float *v_h = m->v_cache[l] + k * KV_DIM + kvh * HEAD_DIM;
                for (int d = 0; d < HEAD_DIM; d++)
                    attn[qh * HEAD_DIM + d] += wt * v_h[d];
            }
        }

        matmul_t(h, attn, m->attn_out[l], HIDDEN_DIM, HIDDEN_DIM);
        for (int i = 0; i < HIDDEN_DIM; i++) h[i] += residual[i];

        /* FFN (SwiGLU) */
        rmsnorm(residual, h, (const float *)m->ffn_norm[l]->ptr, HIDDEN_DIM, 1e-6f);
        matmul_t(m->tmp_gate, residual, m->ffn_gate[l], HIDDEN_DIM, FFN_DIM);
        matmul_t(m->tmp_up,   residual, m->ffn_up[l],   HIDDEN_DIM, FFN_DIM);
        for (int j = 0; j < FFN_DIM; j++) {
            float g = m->tmp_gate[j];
            m->tmp_ffn[j] = (g / (1.0f + expf(-g))) * m->tmp_up[j];
        }
        matmul_t(m->tmp_ffn_out, m->tmp_ffn, m->ffn_down[l], FFN_DIM, HIDDEN_DIM);
        for (int i = 0; i < HIDDEN_DIM; i++) h[i] += m->tmp_ffn_out[i];

        if (tl_trace && cur == m->n_prompt - 1) {
            float s = 0.0f;
            for (int i = 0; i < HIDDEN_DIM; i++) s += h[i] * h[i];
            printf("layer %d |h|=%.3f\n", l, sqrtf(s));
        }
    }

    rmsnorm(h, h, (const float *)m->w_output_norm->ptr, HIDDEN_DIM, 1e-6f);
    matmul_f32(m->logits, h, m->w_output_f32, HIDDEN_DIM, VOCAB_SIZE);
}

/* ================================================================
 *  SAMPLING (top-k + temperature; temp<=0 -> greedy)
 * ================================================================ */
static int sample(float *logits, int vocab, int top_k, float temp) {
    if (temp <= 1e-5f) {
        int best = 0;
        for (int i = 1; i < vocab; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    if (top_k <= 0 || top_k > 256) top_k = 256;
    if (top_k > vocab) top_k = vocab;

    float maxv = logits[0];
    for (int i = 1; i < vocab; i++) if (logits[i] > maxv) maxv = logits[i];

    float top_vals[256];
    int   top_idx[256];
    for (int i = 0; i < top_k; i++) { top_vals[i] = -1e30f; top_idx[i] = 0; }

    for (int i = 0; i < vocab; i++) {
        float v = (logits[i] - maxv) / temp;
        for (int j = 0; j < top_k; j++) {
            if (v > top_vals[j]) {
                for (int k2 = top_k - 1; k2 > j; k2--) {
                    top_vals[k2] = top_vals[k2-1];
                    top_idx[k2]  = top_idx[k2-1];
                }
                top_vals[j] = v;
                top_idx[j]  = i;
                break;
            }
        }
    }

    float mx = top_vals[0], sum = 0.0f;
    for (int i = 0; i < top_k; i++) { top_vals[i] = expf(top_vals[i] - mx); sum += top_vals[i]; }
    float r = ((float)rand() / (float)RAND_MAX) * sum;
    float cum = 0.0f;
    for (int i = 0; i < top_k; i++) {
        cum += top_vals[i];
        if (r <= cum) return top_idx[i];
    }
    return top_idx[top_k - 1];
}

/* ================================================================
 *  PUBLIC API (native + WASM)
 * ================================================================ */

/* Load model from an in-memory GGUF buffer (we keep the pointer).
 * size is int (not int64) so JS can pass it without BigInt (< 2 GB). */
TL_EXPORT int tl_init(uint8_t *buf, int size) {
    model_load(&g_model, buf, (int64_t)size);
    return 0;
}

/* Tokenize prompt and run prefill. Returns prompt token count. */
TL_EXPORT int tl_set_prompt(const char *text) {
    Model *m = &g_model;
    m->n_tokens = 0;
    m->tokens[m->n_tokens++] = 1; /* BOS */
    int n = tokenize(text, m->tokens + m->n_tokens, MAX_TOKENS - m->n_tokens - 1);
    m->n_tokens += n;
    m->n_prompt = m->n_tokens;
    /* prefill: forward over all prompt tokens */
    for (int i = 1; i <= m->n_tokens; i++) {
        int save = m->n_tokens;
        m->n_tokens = i;
        model_forward(m);
        m->n_tokens = save;
    }
    return m->n_prompt;
}

/* Generate one token. Samples from the logits already computed for the
 * current position, appends the token, then runs the forward pass on it
 * so logits for the next step are ready. Returns token id, or -1 (EOS/limit). */
TL_EXPORT int tl_step(float temp, int top_k) {
    Model *m = &g_model;
    if (m->n_tokens >= MAX_TOKENS - 1 || m->n_tokens >= MAX_SEQ) return -1;
    int next = sample(m->logits, g_vocab_size, top_k, temp);
    if (next == 2) return -1; /* EOS */
    m->tokens[m->n_tokens] = next;
    m->n_tokens++;
    model_forward(m); /* logits for the following token */
    return next;
}

/* Decode a token id to text (U+2581 -> space). Static buffer. */
TL_EXPORT const char *tl_token_str(int id) {
    return decode_token(id);
}

/* Number of tokens generated so far (excluding prompt). */
TL_EXPORT int tl_generated(void) {
    return g_model.n_tokens - g_model.n_prompt;
}

/* ================================================================
 *  NATIVE MAIN
 * ================================================================ */
#ifndef __EMSCRIPTEN__
int main(int argc, char **argv) {
    const char *model_path = "model/tinillama.gguf";
    const char *prompt = "The secret to happiness is";
    int max_new = 30;
    float temp = 0.8f;
    int top_k = 25;

    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) prompt = argv[2];
    if (argc >= 4) max_new = atoi(argv[3]);
    if (argc >= 5) temp = (float)atof(argv[4]);
    if (argc >= 6) top_k = atoi(argv[5]);

    FILE *f = fopen(model_path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", model_path); return 1; }
    fseek(f, 0, SEEK_END);
    int64_t size = (int64_t)_ftelli64(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc(size);
    if (!buf) { fprintf(stderr, "malloc failed\n"); return 1; }
    if ((int64_t)fread(buf, 1, size, f) != size) { fprintf(stderr, "short read\n"); return 1; }
    fclose(f);

    printf("Loading %s (%.1f MB)\n", model_path, (double)size / (1024*1024));
    tl_init(buf, size);

    srand((unsigned)time(NULL));
    clock_t t0 = clock();
    int np = tl_set_prompt(prompt);
    double tp = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("Prompt: '%s' (%d tokens, prefill %.1fs)\n", prompt, np, tp);
    printf("Token ids:");
    for (int i = 0; i < g_model.n_prompt; i++) printf(" %d", g_model.tokens[i]);
    printf("\nTop-5 logits after prefill:");
    {
        int ti[5] = {0}; float tv[5] = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
        for (int i = 0; i < VOCAB_SIZE; i++) {
            for (int j = 0; j < 5; j++) if (g_model.logits[i] > tv[j]) {
                for (int k = 4; k > j; k--) { tv[k] = tv[k-1]; ti[k] = ti[k-1]; }
                tv[j] = g_model.logits[i]; ti[j] = i;
                break;
            }
        }
        for (int j = 0; j < 5; j++) printf(" [%d]='%s' %.3f", ti[j], decode_token(ti[j]), tv[j]);
    }
    printf("\n---\n");

    int gen = 0;
    for (int s = 0; s < max_new; s++) {
        int id = tl_step(temp, top_k);
        if (id < 0) break;
        fputs(tl_token_str(id), stdout);
        fflush(stdout);
        gen++;
    }
    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("\n---\n%d tokens in %.1fs (%.2f tok/s)\n", gen, elapsed, gen / elapsed);
    return 0;
}
#endif
