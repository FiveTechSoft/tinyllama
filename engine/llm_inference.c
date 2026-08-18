/*
 * llm_inference.c - Motor de inferencia puro C para TinyLlama GGUF
 * Sin dependencias externas. Solo stdio, stdlib, string, stdint, math.
 *
 * Compilar (Windows):
 *   gcc -O2 -o llm_inference.exe llm_inference.c -lm
 *
 * Ejecutar:
 *   llm_inference.exe C:/tmp/llm_pequeno.gguf "DREAM" 30
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

/* Force optimal thread count for this machine */
static void __attribute__((constructor)) set_threads(void) {
    omp_set_num_threads(5);
}

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
#define MAX_TOKENS    512

/* ================================================================
 *  HALF-PRECISION FLOAT (F16 -> F32)
 *  IEEE 754 half-precision manual decoder
 * ================================================================ */
static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)(h >> 15)) & 0x1;
    uint32_t exp  = ((uint32_t)(h >> 10)) & 0x1F;
    uint32_t man  = (uint32_t)h & 0x3FF;
    uint32_t result;

    if (exp == 0) {
        if (man == 0) { result = sign << 31; }
        else {
            exp = 1;
            while ((man & 0x400) == 0 && exp > 0) { man <<= 1; exp--; }
            man &= 0x3FF;
            result = (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        result = (sign << 31) | (0xFF << 23) | (man << 13);
    } else {
        result = (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &result, 4);
    return f;
}

/* ================================================================
 *  FILE SIZE UTILITY
 * ================================================================ */
static long file_size(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long s = ftell(f);
    fclose(f);
    return s;
}

static int64_t file_size64(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    _fseeki64(f, 0, SEEK_END);
    int64_t s = _ftelli64(f);
    fclose(f);
    return s;
}

/* ================================================================
 *  GGUF READER (minimal, reads into memory)
 * ================================================================ */
typedef struct {
    uint8_t *data;
    size_t size;
    size_t pos;
    int version;
    int64_t num_tensors;
    int64_t num_kv;
    int num_meta;
    char meta_keys[64][48];
    char meta_vals[64][128];
    int num_tfound;
    char tnames[128][64];
    int64_t toffsets[128];
} GGUF;

static uint64_t read_u64_buf(const uint8_t *buf, size_t *pos) {
    uint64_t v;
    memcpy(&v, buf + *pos, 8);
    *pos += 8;
    return v;
}

static uint32_t read_u32_buf(const uint8_t *buf, size_t *pos) {
    uint32_t v;
    memcpy(&v, buf + *pos, 4);
    *pos += 4;
    return v;
}

static void read_string_buf(const uint8_t *buf, size_t *pos, char *out, int maxlen) {
    uint64_t len = read_u64_buf(buf, pos);
    if (len >= (uint64_t)maxlen) len = maxlen - 1;
    memcpy(out, buf + *pos, len);
    out[len] = '\0';
    *pos += len;
}

static void gguf_load(GGUF *g, const char *path) {
    g->size = file_size(path);
    printf("Loading '%s' (%ld bytes, %.1f MB)\n", path, g->size, (double)g->size / (1024*1024));
    if (g->size < 0) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    g->data = malloc(g->size);
    if (!g->data) { fprintf(stderr, "malloc failed\n"); exit(1); }
    FILE *f = fopen(path, "rb");
    fread(g->data, 1, g->size, f);
    fclose(f);
    g->pos = 0;

    /* Check magic */
    if (memcmp(g->data, "GGUF", 4) != 0) {
        fprintf(stderr, "Not a GGUF file\n"); exit(1);
    }
    g->pos = 4;

    g->version = read_u32_buf(g->data, &g->pos);
    g->num_tensors = read_u64_buf(g->data, &g->pos);
    g->num_kv = read_u64_buf(g->data, &g->pos);

    /* Parse metadata */
    g->num_meta = 0;
    for (int64_t i = 0; i < g->num_kv && g->num_meta < 64; i++) {
        char key[48], val[128];
        read_string_buf(g->data, &g->pos, key, 48);
        uint64_t vtype = read_u64_buf(g->data, &g->pos);
        uint64_t vlen = read_u64_buf(g->data, &g->pos);
        if (vlen >= 128) vlen = 127;
        memcpy(val, g->data + g->pos, vlen);
        val[vlen] = '\0';
        g->pos += vlen;
        (void)vtype;
        if (g->num_meta < 64) {
            strncpy(g->meta_keys[g->num_meta], key, 47);
            g->meta_keys[g->num_meta][47] = '\0';
            strncpy(g->meta_vals[g->num_meta], val, 127);
            g->meta_vals[g->num_meta][127] = '\0';
            g->num_meta++;
        }
    }

    /* Parse tensor index */
    g->num_tfound = 0;
    for (int64_t i = 0; i < g->num_tensors && g->num_tfound < 128; i++) {
        char tname[64];
        read_string_buf(g->data, &g->pos, tname, 64);
        uint32_t ttype = read_u32_buf(g->data, &g->pos);
        uint32_t ndims = read_u32_buf(g->data, &g->pos);
        int64_t dims[8] = {0};
        for (uint32_t d = 0; d < ndims && d < 8; d++) {
            dims[d] = (int64_t)read_u64_buf(g->data, &g->pos);
        }
        int64_t offset = (int64_t)read_u64_buf(g->data, &g->pos);
        (void)ttype;

        memset(g->tnames[g->num_tfound], 0, 64);
        strncpy(g->tnames[g->num_tfound], tname, 63);
        g->toffsets[g->num_tfound] = offset;
        g->num_tfound++;
    }
}

static int gguf_find(GGUF *g, const char *name) {
    for (int i = 0; i < g->num_tfound; i++) {
        if (strcmp(g->tnames[i], name) == 0) return i;
    }
    return -1;
}

static void load_tensor_f32(GGUF *g, int idx, float *out) {
    const uint8_t *d = g->data + g->toffsets[idx];
    int64_t nelem = 1;
    /* We need to figure out nelem... */
    /* Re-parse tensor at index idx to get dims */
    /* For simplicity, we'll load all tensors we need by scanning */
    /* Actually, nelems are known statically from architecture */
    fprintf(stderr, "ERROR: should use static nelem\n");
    exit(1);
}

/* Better approach: store nelems directly */
typedef struct {
    char name[64];
    int64_t offset;
    int64_t nelem;
    uint32_t type;  /* ggml_type: 0=F32, 1=F16 */
} TensorInfo;

typedef struct {
    uint8_t *data;
    int64_t size;
    size_t pos;
    int version;
    int num_meta;
    char meta_keys[64][48];
    char meta_vals[64][128];
    TensorInfo tensors[256];
    int num_tensors;
} GGUFReader;

static void ggufr_load(GGUFReader *r, const char *path) {
    r->pos = 0;
    r->size = file_size64(path);
    printf("Loading: %s (%lld bytes)\n", path, r->size);
    fflush(stdout);
    r->data = malloc(r->size);
    if (!r->data) { fprintf(stderr, "malloc failed for %lld bytes\n", r->size); exit(1); }
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    size_t read = fread(r->data, 1, r->size, f);
    fclose(f);
    if ((int64_t)read != r->size) { fprintf(stderr, "Short read: %zu vs %lld\n", read, r->size); exit(1); }
    size_t p = 0;

    if (memcmp(r->data, "GGUF", 4) != 0) { fprintf(stderr, "Bad magic\n"); exit(1); }
    p = 4;

    r->version = (int)read_u32_buf(r->data, &p);
    int64_t tcount = (int64_t)read_u64_buf(r->data, &p);
    int64_t kvcount = (int64_t)read_u64_buf(r->data, &p);

    r->num_meta = 0;
    for (int64_t i = 0; i < kvcount && r->num_meta < 64; i++) {
        char key[48];
        read_string_buf(r->data, &p, key, 48);
        uint32_t vtype = read_u32_buf(r->data, &p);

        if (i < 3) printf("  KV[%lld] key='%s' vtype=%u pos=%zu\n", (long long)i, key, vtype, p);

        /* GGUF v3 type sizes: 0=U8 1=I8 2=U16 3=I16 4=U32 5=I32
         * 6=F32 7=BOOL 8=STRING 9=ARRAY 10=U64 11=I64 12=F64 */
        static const int gguf_type_sz[13] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
        if (vtype == 8) {
            /* STRING */
            uint64_t slen = read_u64_buf(r->data, &p);
            char val[128];
            uint64_t cplen = slen < 127 ? slen : 127;
            memcpy(val, r->data + p, cplen);
            val[cplen] = '\0';
            p += slen;
            if (r->num_meta < 64) {
                strncpy(r->meta_keys[r->num_meta], key, 47);
                r->meta_keys[r->num_meta][47] = '\0';
                strncpy(r->meta_vals[r->num_meta], val, 127);
                r->meta_vals[r->num_meta][127] = '\0';
                r->num_meta++;
            }
        } else if (vtype == 9) {
            /* ARRAY: skip element type + count + all elements */
            uint32_t atype = read_u32_buf(r->data, &p);
            uint64_t alen = read_u64_buf(r->data, &p);
            for (uint64_t a = 0; a < alen; a++) {
                if (atype == 8) {
                    uint64_t sl = read_u64_buf(r->data, &p);
                    p += sl;
                } else {
                    p += gguf_type_sz[atype < 13 ? atype : 0];
                }
            }
        } else {
            p += gguf_type_sz[vtype < 13 ? vtype : 0];
        }
    }

    printf("  %d metadata keys\n", r->num_meta);
    printf("  KV loop ended at offset %zu\n", p);

    r->num_tensors = 0;
    for (int64_t i = 0; i < tcount && r->num_tensors < 256; i++) {
        read_string_buf(r->data, &p, r->tensors[r->num_tensors].name, 64);
        uint32_t ndims = read_u32_buf(r->data, &p);
        int64_t nelem = 1;
        for (uint32_t d = 0; d < ndims && d < 8; d++) {
            int64_t dim = (int64_t)read_u64_buf(r->data, &p);
            nelem *= dim;
        }
        uint32_t ttype = read_u32_buf(r->data, &p);
        int64_t offset = (int64_t)read_u64_buf(r->data, &p);
        r->tensors[r->num_tensors].offset = offset;
        r->tensors[r->num_tensors].nelem = nelem;
        r->tensors[r->num_tensors].type = ttype;
        if (r->num_tensors < 5 || strcmp(r->tensors[r->num_tensors].name, "blk.0.attn_norm.weight") == 0) {
            printf("  [%d] '%s' raw_offset=%lld nelem=%lld\n", r->num_tensors,
                   r->tensors[r->num_tensors].name, (long long)offset, (long long)nelem);
        }
        r->num_tensors++;
    }

    /* Data section starts after tensor index, aligned to 32 bytes */
    int64_t data_start = ((p + 31) / 32) * 32;
    printf("  Tensor index ends at %zu, data_start=%lld\n", p, (long long)data_start);
    for (int i = 0; i < r->num_tensors; i++) {
        printf("  [%d] '%s' final_offset=%lld (raw=%lld + data_start=%lld)\n",
               i, r->tensors[i].name, (long long)(r->tensors[i].offset + data_start),
               (long long)r->tensors[i].offset, (long long)data_start);
        r->tensors[i].offset += data_start;
    }

    printf("  %d tensors loaded\n", r->num_tensors);
    if (r->num_tensors > 0) {
        printf("  First: '%s' nelem=%lld offset=%lld\n",
               r->tensors[0].name, (long long)r->tensors[0].nelem, (long long)r->tensors[0].offset);
    }
}

static int ggufr_find(GGUFReader *r, const char *name) {
    for (int i = 0; i < r->num_tensors; i++) {
        if (strcmp(r->tensors[i].name, name) == 0) return i;
    }
    return -1;
}

static float *ggufr_load_tensor(GGUFReader *r, int idx) {
    float *out = malloc(r->tensors[idx].nelem * sizeof(float));
    const uint8_t *raw = r->data + r->tensors[idx].offset;
    int64_t n = r->tensors[idx].nelem;
    uint32_t ttype = r->tensors[idx].type;
    if (ttype == 0) {
        /* F32: 4 bytes per element */
        memcpy(out, raw, n * sizeof(float));
    } else {
        /* F16 (or other): 2 bytes per element, convert to F32 */
        for (int64_t i = 0; i < n; i++) {
            uint16_t h;
            memcpy(&h, raw + i * 2, 2);
            out[i] = f16_to_f32(h);
        }
    }
    return out;
}

static void ggufr_free(GGUFReader *r) {
    free(r->data);
}

/* ================================================================
 *  TOKENIZER: Read vocab from GGUF, BPE encode
 * ================================================================ */
static void load_vocab(GGUFReader *r, char **vocab, int *vocab_size) {
    /* Search raw GGUF data for tokenizer.ggml.tokens key */
    size_t p = 0;
    uint8_t *d = r->data;
    int64_t tcount = (int64_t)((uint64_t)d[8] | (uint64_t)d[9]<<8 | (uint64_t)d[10]<<16 | (uint64_t)d[11]<<24 |
                               (uint64_t)d[12]<<32 | (uint64_t)d[13]<<40 | (uint64_t)d[14]<<48 | (uint64_t)d[15]<<56);
    int64_t kvcount = (int64_t)((uint64_t)d[16] | (uint64_t)d[17]<<8 | (uint64_t)d[18]<<16 | (uint64_t)d[19]<<24 |
                                (uint64_t)d[20]<<32 | (uint64_t)d[21]<<40 | (uint64_t)d[22]<<48 | (uint64_t)d[23]<<56);
    p = 24;

    static const int gguf_type_sz[13] = {1,1,2,2,4,4,4,1,0,0,8,8,8};

    for (int64_t i = 0; i < kvcount; i++) {
        /* Read key string */
        uint64_t klen = 0;
        memcpy(&klen, d + p, 8); p += 8;
        char key[128];
        uint64_t cplen = klen < 127 ? klen : 127;
        memcpy(key, d + p, cplen);
        key[cplen] = '\0';
        p += klen;

        uint32_t vtype = 0;
        memcpy(&vtype, d + p, 4); p += 4;

        if (strcmp(key, "tokenizer.ggml.tokens") == 0 && vtype == 9) {
            /* ARRAY */
            uint32_t atype = 0;
            memcpy(&atype, d + p, 4); p += 4;
            uint64_t alen = 0;
            memcpy(&alen, d + p, 8); p += 8;

            *vocab_size = (int)alen;
            printf("  Vocab size: %d (atype=%u)\n", *vocab_size, atype);

            for (uint64_t a = 0; a < alen && (int)a < *vocab_size; a++) {
                if (atype == 8) {
                    /* STRING */
                    uint64_t sl = 0;
                    memcpy(&sl, d + p, 8); p += 8;
                    vocab[a] = malloc((int)sl + 1);
                    memcpy(vocab[a], d + p, (int)sl);
                    vocab[a][(int)sl] = '\0';
                    p += sl;
                } else {
                    p += gguf_type_sz[atype < 13 ? atype : 0];
                    vocab[a] = strdup("?");
                }
            }
            printf("  Loaded %d tokens\n", *vocab_size);
            return;
        }

        /* Skip value */
        if (vtype == 8) {
            uint64_t slen = 0;
            memcpy(&slen, d + p, 8); p += 8;
            p += slen;
        } else if (vtype == 9) {
            uint32_t at = 0;
            memcpy(&at, d + p, 4); p += 4;
            uint64_t al = 0;
            memcpy(&al, d + p, 8); p += 8;
            for (uint64_t a = 0; a < al; a++) {
                if (at == 8) {
                    uint64_t sl = 0;
                    memcpy(&sl, d + p, 8); p += 8;
                    p += sl;
                } else {
                    p += gguf_type_sz[at < 13 ? at : 0];
                }
            }
        } else {
            p += gguf_type_sz[vtype < 13 ? vtype : 0];
        }
    }
    *vocab_size = 0;
}

/* BPE tokenizer: encode text into token IDs */
static int tokenize(const char *text, char **vocab, int vocab_size, int *tokens, int max_tokens) {
    int n_tokens = 0;

    /* BPE: start with byte tokens, then iteratively merge lowest-ID pair */
    int ids[4096];
    char pieces[4096][128];
    int n = 0;

    /* Initial tokenization: map each byte to its token */
    /* In Llama tokenizer: bytes 0-255 map to tokens 3-258 via <0xXX> format */
    /* But we can also match single-char tokens from vocab */
    for (int i = 0; text[i] && n < 4090; i++) {
        unsigned char b = (unsigned char)text[i];
        ids[n] = b + 3;  /* default: byte token */
        snprintf(pieces[n], 128, "<0x%02X>", b);
        /* Check if there's a single-char token in vocab that matches */
        char single[2] = {(char)b, 0};
        for (int v = 259; v < vocab_size; v++) {
            if (vocab[v] && strcmp(vocab[v], single) == 0) {
                ids[n] = v;
                strcpy(pieces[n], single);
                break;
            }
        }
        n++;
    }

    /* BPE merges: iteratively find and merge lowest-ID pair */
    for (int iter = 0; iter < 4096; iter++) {
        int best_id = -1, best_pos = -1;
        int best_rank = vocab_size + 1;

        for (int i = 0; i < n - 1; i++) {
            /* Construct merged piece */
            char merged[256];
            snprintf(merged, sizeof(merged), "%s%s", pieces[i], pieces[i + 1]);

            /* Search vocab for this merged piece */
            for (int v = 259; v < vocab_size; v++) {
                if (vocab[v] && strcmp(vocab[v], merged) == 0) {
                    if (v < best_rank) {
                        best_rank = v;
                        best_id = v;
                        best_pos = i;
                    }
                    break;
                }
            }
        }

        if (best_pos < 0) break;  /* no more merges */

        /* Merge: replace [best_pos, best_pos+1] with best_id */
        ids[best_pos] = best_id;
        snprintf(pieces[best_pos], sizeof(pieces[best_pos]), "%s", ""); 
        /* Rebuild pieces from vocab */
        if (vocab[best_id]) {
            strncpy(pieces[best_pos], vocab[best_id], 127);
            pieces[best_pos][127] = '\0';
        }
        /* Shift remaining */
        for (int i = best_pos + 1; i < n - 1; i++) {
            ids[i] = ids[i + 1];
            strcpy(pieces[i], pieces[i + 1]);
        }
        n--;
    }

    /* Copy result */
    n_tokens = n < max_tokens ? n : max_tokens;
    memcpy(tokens, ids, n_tokens * sizeof(int));
    return n_tokens;
}

/* ================================================================
 *  MODEL STRUCTURE
 * ================================================================ */
typedef struct {
    GGUFReader gguf;

    float *w_emb;
    float *w_emb_norm;
    float *w_output_norm;
    float *w_output;

    float *attn_norm[NUM_LAYERS];
    float *attn_q[NUM_LAYERS];    /* [HIDDEN_DIM x HIDDEN_DIM] */
    float *attn_k[NUM_LAYERS];    /* [KV_DIM x HIDDEN_DIM] */
    float *attn_v[NUM_LAYERS];    /* [KV_DIM x HIDDEN_DIM] */
    float *attn_out[NUM_LAYERS];  /* [HIDDEN_DIM x HIDDEN_DIM] */

    float *ffn_norm[NUM_LAYERS];
    float *ffn_gate[NUM_LAYERS];
    float *ffn_up[NUM_LAYERS];
    float *ffn_down[NUM_LAYERS];

    /* KV cache: [NUM_LAYERS][MAX_SEQ][KV_DIM] */
    float *k_cache[NUM_LAYERS];
    float *v_cache[NUM_LAYERS];
    int seq_len;

    /* Pre-allocated temp buffers (eliminates malloc per token) */
    float *tmp_h;
    float *tmp_residual;
    float *tmp_Q_proj;
    float *tmp_K_proj;
    float *tmp_V_proj;
    float *tmp_attn_result;
    float *tmp_gate_v;
    float *tmp_up_v;
    float *tmp_ffn_buf;
    float *tmp_ffn_result;

    /* Temp buffers */
    float *buf;      /* large contiguous buffer for all temp storage */
    int buf_cap;
    int trace_heads;

    /* Tokenizer */
    char *vocab[VOCAB_SIZE];  /* token strings */
    int vocab_size;
} Model;

static int cfg_int(GGUFReader *r, const char *key, int default_val) {
    for (int i = 0; i < r->num_meta; i++) {
        if (strcmp(r->meta_keys[i], key) == 0) {
            return atoi(r->meta_vals[i]);
        }
    }
    return default_val;
}

static void model_load(Model *m, const char *path) {
    ggufr_load(&m->gguf, path);

    int hidden = cfg_int(&m->gguf, "general.embedding_length", HIDDEN_DIM);
    int layers = cfg_int(&m->gguf, "general.block_count", NUM_LAYERS);
    int heads  = cfg_int(&m->gguf, "general.attention.head_count", NUM_HEADS);
    int kvh    = cfg_int(&m->gguf, "general.attention.head_count_kv", NUM_KV_HEADS);
    int head_d = cfg_int(&m->gguf, "general.attention.head_dim", HEAD_DIM);
    int vocab  = cfg_int(&m->gguf, "general.vocab_size", VOCAB_SIZE);
    int ffn    = cfg_int(&m->gguf, "general.feed_forward_length", FFN_DIM);

    printf("  Config: hidden=%d layers=%d heads=%d kv_heads=%d head_dim=%d ffn=%d vocab=%d\n",
           hidden, layers, heads, kvh, head_d, ffn, vocab);

    printf("  Loading weights...\n");
    fflush(stdout);
    clock_t t0 = clock();

    /* Load tokenizer vocabulary */
    load_vocab(&m->gguf, m->vocab, &m->vocab_size);
    printf("  Vocab loaded: %d tokens\n", m->vocab_size);

    int idx;

    idx = ggufr_find(&m->gguf, "token_embd.weight");
    if (idx < 0) { fprintf(stderr, "tensor 'token_embd.weight' not found\n"); exit(1); }
    m->w_emb = ggufr_load_tensor(&m->gguf, idx);

    /* TinyLlama has no embedding norm */
    m->w_emb_norm = NULL;

    idx = ggufr_find(&m->gguf, "output_norm.weight");
    if (idx < 0) { fprintf(stderr, "tensor 'output_norm.weight' not found\n"); exit(1); }
    m->w_output_norm = ggufr_load_tensor(&m->gguf, idx);

    idx = ggufr_find(&m->gguf, "output.weight");
    if (idx < 0) { fprintf(stderr, "tensor 'output.weight' not found\n"); exit(1); }
    m->w_output = ggufr_load_tensor(&m->gguf, idx);

    for (int l = 0; l < layers; l++) {
        char name[128];

        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", l);
        idx = ggufr_find(&m->gguf, name);
        if (idx < 0) { fprintf(stderr, "tensor '%s' not found\n", name); exit(1); }
        m->attn_norm[l] = ggufr_load_tensor(&m->gguf, idx);

        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
        idx = ggufr_find(&m->gguf, name);
        if (idx < 0) { fprintf(stderr, "tensor '%s' not found\n", name); exit(1); }
        m->attn_q[l] = ggufr_load_tensor(&m->gguf, idx);

        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);
        idx = ggufr_find(&m->gguf, name);
        if (idx < 0) { fprintf(stderr, "tensor '%s' not found\n", name); exit(1); }
        m->attn_k[l] = ggufr_load_tensor(&m->gguf, idx);

        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", l);
        idx = ggufr_find(&m->gguf, name);
        if (idx < 0) { fprintf(stderr, "tensor '%s' not found\n", name); exit(1); }
        m->attn_v[l] = ggufr_load_tensor(&m->gguf, idx);

        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);
        idx = ggufr_find(&m->gguf, name);
        if (idx < 0) { fprintf(stderr, "tensor '%s' not found\n", name); exit(1); }
        m->attn_out[l] = ggufr_load_tensor(&m->gguf, idx);

        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", l);
        idx = ggufr_find(&m->gguf, name);
        if (idx < 0) { fprintf(stderr, "tensor '%s' not found\n", name); exit(1); }
        m->ffn_norm[l] = ggufr_load_tensor(&m->gguf, idx);

        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", l);
        idx = ggufr_find(&m->gguf, name);
        if (idx < 0) { fprintf(stderr, "tensor '%s' not found\n", name); exit(1); }
        m->ffn_gate[l] = ggufr_load_tensor(&m->gguf, idx);

        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", l);
        idx = ggufr_find(&m->gguf, name);
        if (idx < 0) { fprintf(stderr, "tensor '%s' not found\n", name); exit(1); }
        m->ffn_up[l] = ggufr_load_tensor(&m->gguf, idx);

        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", l);
        idx = ggufr_find(&m->gguf, name);
        if (idx < 0) { fprintf(stderr, "tensor '%s' not found\n", name); exit(1); }
        m->ffn_down[l] = ggufr_load_tensor(&m->gguf, idx);
    }

    /* Allocate KV cache */
    for (int l = 0; l < NUM_LAYERS; l++) {
        m->k_cache[l] = calloc(MAX_SEQ * KV_DIM, sizeof(float));
        m->v_cache[l] = calloc(MAX_SEQ * KV_DIM, sizeof(float));
    }
    m->seq_len = 0;

    /* Pre-allocate temp buffers (reused every forward pass) */
    m->tmp_h = malloc(HIDDEN_DIM * sizeof(float));
    m->tmp_residual = malloc(HIDDEN_DIM * sizeof(float));
    m->tmp_Q_proj = malloc(HIDDEN_DIM * sizeof(float));
    m->tmp_K_proj = malloc(KV_DIM * sizeof(float));
    m->tmp_V_proj = malloc(KV_DIM * sizeof(float));
    m->tmp_attn_result = malloc(HIDDEN_DIM * sizeof(float));
    m->tmp_gate_v = malloc(FFN_DIM * sizeof(float));
    m->tmp_up_v = malloc(FFN_DIM * sizeof(float));
    m->tmp_ffn_buf = malloc(FFN_DIM * sizeof(float));
    m->tmp_ffn_result = malloc(HIDDEN_DIM * sizeof(float));

    printf("  Done in %.2f seconds\n", (double)(clock() - t0) / CLOCKS_PER_SEC);
}

static void model_free(Model *m) {
    free(m->w_emb); free(m->w_emb_norm);
    free(m->w_output_norm); free(m->w_output);
    for (int i = 0; i < m->vocab_size; i++) free(m->vocab[i]);
    for (int l = 0; l < NUM_LAYERS; l++) {
        free(m->attn_norm[l]); free(m->attn_q[l]); free(m->attn_k[l]);
        free(m->attn_v[l]); free(m->attn_out[l]);
        free(m->ffn_norm[l]); free(m->ffn_gate[l]);
        free(m->ffn_up[l]); free(m->ffn_down[l]);
        free(m->k_cache[l]); free(m->v_cache[l]);
    }
    free(m->tmp_h); free(m->tmp_residual);
    free(m->tmp_Q_proj); free(m->tmp_K_proj); free(m->tmp_V_proj);
    free(m->tmp_attn_result);
    free(m->tmp_gate_v); free(m->tmp_up_v);
    free(m->tmp_ffn_buf); free(m->tmp_ffn_result);
    ggufr_free(&m->gguf);
}

/* ================================================================
 *  TRANSFORMER OPERATIONS
 * ================================================================ */

static void rmsnorm(float *out, const float *x, const float *w, int dim, float eps) {
    float sum = 0.0f;
    int i;
#if defined(__AVX2__)
    __m256 vsum = _mm256_setzero_ps();
    for (i = 0; i + 8 <= dim; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        vsum = _mm256_add_ps(vsum, _mm256_mul_ps(vx, vx));
    }
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    __m128 lo = _mm256_castps256_ps128(vsum);
    __m128 s4 = _mm_add_ps(lo, hi);
    __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    sum = _mm_cvtss_f32(s1);
    for (; i < dim; i++) sum += x[i] * x[i];
#elif defined(__SSE2__)
    __m128 vsum = _mm_setzero_ps();
    for (i = 0; i + 4 <= dim; i += 4) {
        __m128 vx = _mm_loadu_ps(x + i);
        vsum = _mm_add_ps(vsum, _mm_mul_ps(vx, vx));
    }
    float tmp[4];
    _mm_storeu_ps(tmp, vsum);
    sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < dim; i++) sum += x[i] * x[i];
#else
    for (i = 0; i < dim; i++) sum += x[i] * x[i];
#endif
    float scale = 1.0f / sqrtf(sum / dim + eps);
    for (i = 0; i < dim; i++) out[i] = x[i] * scale * w[i];
}

static inline float dot_product(const float *a, const float *b, int n) {
#if defined(__AVX2__)
    int i = 0;
    __m256 sum = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
    }
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 s4 = _mm_add_ps(lo, hi);
    __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    float result = _mm_cvtss_f32(s1);
    for (; i < n; i++) result += a[i] * b[i];
    return result;
#elif defined(__SSE2__)
    int i = 0;
    __m128 sum = _mm_setzero_ps();
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
    }
    float tmp[4];
    _mm_storeu_ps(tmp, sum);
    float result = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; i++) result += a[i] * b[i];
    return result;
#else
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
#endif
}

static void matmul(float *out, const float *x, const float *W, int in_dim, int out_dim) {
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < out_dim; j++) {
        out[j] = dot_product(x, W + j * in_dim, in_dim);
    }
}

/* Pre-computed RoPE table: cos/sin for all positions × head_dim */
static float rope_cos[MAX_SEQ][HEAD_DIM];
static float rope_sin[MAX_SEQ][HEAD_DIM];
static int rope_initialized = 0;

static void rope_init(void) {
    for (int pos = 0; pos < MAX_SEQ; pos++) {
        for (int i = 0; i < HEAD_DIM; i += 2) {
            float freq = 1.0f / powf(10000.0f, (float)i / HEAD_DIM);
            float angle = pos * freq;
            rope_cos[pos][i] = rope_cos[pos][i+1] = cosf(angle);
            rope_sin[pos][i] = rope_sin[pos][i+1] = sinf(angle);
        }
    }
    rope_initialized = 1;
}

static void apply_rope_single(float *x, int pos, int head_dim) {
    if (!rope_initialized) rope_init();
    for (int i = 0; i < head_dim; i += 2) {
        float c = rope_cos[pos][i];
        float s = rope_sin[pos][i];
        float x0 = x[i];
        float x1 = x[i + 1];
        x[i]     = x0 * c - x1 * s;
        x[i + 1] = x0 * s + x1 * c;
    }
}

static void masked_softmax(float *out, const float *logits, int n, int causal_pos) {
    float maxv = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > maxv) maxv = logits[i];
    if (!isfinite(maxv)) maxv = 0.0f;

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = logits[i] - maxv;
        if (v > 50.0f) v = 50.0f;
        if (v < -50.0f) v = -50.0f;
        out[i] = expf(v);
        if (i > causal_pos) out[i] = 0.0f;
        sum += out[i];
    }
    if (sum > 0.0f) {
        for (int i = 0; i < n; i++) out[i] /= sum;
    } else {
        for (int i = 0; i < n; i++) out[i] = 1.0f / n;
    }
}

/* One token forward with GQA + KV cache. Returns logits for the new token. */
static void model_forward(Model *m, int *tokens, int seq_len, float *logits) {
    int cur = seq_len - 1;
    int tok = tokens[cur];
    if (tok < 0 || tok >= VOCAB_SIZE) tok = 0;

    float *h = m->tmp_h;

    /* Embed */
    memcpy(h, m->w_emb + tok * HIDDEN_DIM, HIDDEN_DIM * sizeof(float));

    /* Emb norm (skip if not present) */
    if (m->w_emb_norm) {
        rmsnorm(h, h, m->w_emb_norm, HIDDEN_DIM, 1e-6f);
    }

    float *residual = m->tmp_residual;
    float *Q_proj = m->tmp_Q_proj;
    float *K_proj = m->tmp_K_proj;
    float *V_proj = m->tmp_V_proj;

    for (int l = 0; l < NUM_LAYERS; l++) {
        memcpy(residual, h, HIDDEN_DIM * sizeof(float));

        /* Attn norm */
        rmsnorm(h, h, m->attn_norm[l], HIDDEN_DIM, 1e-6f);

        /* Q projection: HIDDEN_DIM -> HIDDEN_DIM */
        matmul(Q_proj, h, m->attn_q[l], HIDDEN_DIM, HIDDEN_DIM);

        /* K projection: HIDDEN_DIM -> KV_DIM */
        matmul(K_proj, h, m->attn_k[l], HIDDEN_DIM, KV_DIM);

        /* V projection: HIDDEN_DIM -> KV_DIM */
        matmul(V_proj, h, m->attn_v[l], HIDDEN_DIM, KV_DIM);

        /* Apply RoPE to Q (all heads) and K (kv heads) */
        for (int hd = 0; hd < NUM_HEADS; hd++) {
            apply_rope_single(Q_proj + hd * HEAD_DIM, cur, HEAD_DIM);
        }
        for (int hd = 0; hd < NUM_KV_HEADS; hd++) {
            apply_rope_single(K_proj + hd * HEAD_DIM, cur, HEAD_DIM);
        }

        /* Store K,V in cache */
        memcpy(m->k_cache[l] + cur * KV_DIM, K_proj, KV_DIM * sizeof(float));
        memcpy(m->v_cache[l] + cur * KV_DIM, V_proj, KV_DIM * sizeof(float));

        /* GQA Attention: 32 Q heads, 4 KV heads, 8 Q per KV */
        float *attn_result = m->tmp_attn_result;
        memset(attn_result, 0, HIDDEN_DIM * sizeof(float));
        int groups_per_kv = NUM_HEADS / NUM_KV_HEADS;  /* 8 */

    #pragma omp parallel for schedule(static)
        for (int qh = 0; qh < NUM_HEADS; qh++) {
            int kvh = qh / groups_per_kv;  /* which KV head this Q head uses */
            float *q_h = Q_proj + qh * HEAD_DIM;

            /* Attention scores against all cached K positions */
            float attn_w[MAX_SEQ];
            float attn_logits[MAX_SEQ];
            for (int k = 0; k <= cur; k++) {
                float *k_h = m->k_cache[l] + k * KV_DIM + kvh * HEAD_DIM;
                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) dot += q_h[d] * k_h[d];
                attn_logits[k] = dot / sqrtf((float)HEAD_DIM);
            }
            masked_softmax(attn_w, attn_logits, cur + 1, cur);

            /* Accumulate weighted V values */
            for (int k = 0; k <= cur; k++) {
                float wt = attn_w[k];
                float *v_h = m->v_cache[l] + k * KV_DIM + kvh * HEAD_DIM;
                for (int d = 0; d < HEAD_DIM; d++) {
                    attn_result[qh * HEAD_DIM + d] += wt * v_h[d];
                }
            }
        }

        /* Output projection */
        matmul(h, attn_result, m->attn_out[l], HIDDEN_DIM, HIDDEN_DIM);

        /* Residual */
        for (int i = 0; i < HIDDEN_DIM; i++) h[i] = residual[i] + h[i];

        /* FFN (SwiGLU) — fused gate+up read */
        rmsnorm(residual, h, m->ffn_norm[l], HIDDEN_DIM, 1e-6f);
        float *gate_v = m->tmp_gate_v;
        float *up_v = m->tmp_up_v;
        matmul(gate_v, residual, m->ffn_gate[l], HIDDEN_DIM, FFN_DIM);
        matmul(up_v,   residual, m->ffn_up[l],   HIDDEN_DIM, FFN_DIM);

        /* Fused SiLU gate × up + down projection */
        float *ffn_buf = m->tmp_ffn_buf;
        float *ffn_result = m->tmp_ffn_result;
        for (int j = 0; j < FFN_DIM; j++) {
            float g = gate_v[j];
            float silu = g / (1.0f + expf(-g));
            ffn_buf[j] = silu * up_v[j];
        }
        matmul(ffn_result, ffn_buf, m->ffn_down[l], FFN_DIM, HIDDEN_DIM);

        /* Residual add */
        for (int i = 0; i < HIDDEN_DIM; i++) h[i] += ffn_result[i];
    }

    /* Final norm + output */
    rmsnorm(h, h, m->w_output_norm, HIDDEN_DIM, 1e-6f);
    matmul(logits, h, m->w_output, HIDDEN_DIM, VOCAB_SIZE);

    /* Clamp logits to finite range */
    for (int i = 0; i < VOCAB_SIZE; i++) {
        if (!isfinite(logits[i])) logits[i] = -1e6f;
        else if (logits[i] > 40.0f) logits[i] = 40.0f;
        else if (logits[i] < -40.0f) logits[i] = -40.0f;
    }
}

/* ================================================================
 *  SAMPLING
 * ================================================================ */
static int sample(float *logits, int vocab, int top_k, float temp) {
    if (temp < 1e-10f) temp = 1e-10f;

    float maxv = logits[0];
    for (int i = 1; i < vocab; i++) if (logits[i] > maxv) maxv = logits[i];

    float sum = 0.0f;
    float probs[1024];
    if (top_k > vocab) top_k = vocab;

    float top_vals[256];
    int top_idx[256];
    for (int i = 0; i < top_k; i++) top_vals[i] = -1e30f;

    for (int i = 0; i < vocab; i++) {
        float v = (logits[i] - maxv) / temp;
        for (int j = 0; j < top_k; j++) {
            if (v > top_vals[j]) {
                for (int k2 = top_k - 1; k2 > j; k2--) {
                    top_vals[k2] = top_vals[k2-1];
                    top_idx[k2] = top_idx[k2-1];
                }
                top_vals[j] = v;
                top_idx[j] = i;
                break;
            }
        }
    }

    float mx = top_vals[0];
    for (int i = 0; i < top_k; i++) {
        probs[i] = expf(top_vals[i] - mx);
        sum += probs[i];
    }
    for (int i = 0; i < top_k; i++) probs[i] /= sum;

    float r = (float)rand() / (float)RAND_MAX;
    float cum = 0.0f;
    for (int i = 0; i < top_k; i++) {
        cum += probs[i];
        if (r <= cum) return top_idx[i];
    }
    return top_idx[top_k - 1];
}

/* ================================================================
 *  MAIN
 * ================================================================ */
int main(int argc, char **argv) {
    const char *model_path = "C:/tmp/llm_pequeno.gguf";
    const char *prompt = "DREAM";
    int max_new = 30;
    float temp = 0.8f;
    int top_k = 25;

    int trace_mode = 1;

    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) prompt = argv[2];
    if (argc >= 4) max_new = atoi(argv[3]);
    if (argc >= 5) temp = (float)atof(argv[4]);
    if (argc >= 6) top_k = atoi(argv[5]);

    printf("========================================\n");
    printf("  LLM Motor C Puro + TinyLlama\n");
    printf("  TinyLlama-1.1B | %d layers | %d Q heads, %d KV heads\n", NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS);
    printf("========================================\n\n");

    Model model;
    memset(&model, 0, sizeof(model));
    model.trace_heads = trace_mode;
    model_load(&model, model_path);

    printf("\n--- Generation ---\n");
    printf("Prompt: '%s'\n", prompt);
    printf("Params: max_new=%d temperature=%.1f top_k=%d\n", max_new, temp, top_k);

    /* Tokenize prompt using BPE */
    int prompt_tokens[MAX_TOKENS];
    int n_prompt = 0;
    prompt_tokens[n_prompt++] = 1; /* <s> */
    n_prompt += tokenize(prompt, model.vocab, model.vocab_size,
                         prompt_tokens + n_prompt, MAX_TOKENS - n_prompt - 1);

    printf("Prompt tokens (%d):", n_prompt);
    for (int i = 0; i < n_prompt; i++) printf(" %d", prompt_tokens[i]);
    printf("\n");
    printf("Prompt token strings:");
    for (int i = 0; i < n_prompt; i++) {
        int t = prompt_tokens[i];
        if (t >= 0 && t < model.vocab_size && model.vocab[t])
            printf(" '%s'", model.vocab[t]);
        else
            printf(" [?%d]", t);
    }
    printf("\n\n");

    /* Generate */
    int full_tokens[MAX_TOKENS * 2];
    memcpy(full_tokens, prompt_tokens, n_prompt * sizeof(int));
    int total = n_prompt;

    float logits[VOCAB_SIZE];
    clock_t t0 = clock();

    /* Prefill: process all prompt tokens into KV cache */
    for (int i = 0; i < n_prompt; i++) {
        model_forward(&model, full_tokens, i + 1, logits);
    }

    /* Print top-5 predictions after prefill */
    printf("\nTop-5 logits after prefill:\n");
    {
        int top_idx[5] = {-1,-1,-1,-1,-1};
        float top_val[5] = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
        for (int i = 0; i < VOCAB_SIZE; i++) {
            for (int j = 0; j < 5; j++) {
                if (logits[i] > top_val[j]) {
                    for (int k = 4; k > j; k--) {
                        top_val[k] = top_val[k-1];
                        top_idx[k] = top_idx[k-1];
                    }
                    top_val[j] = logits[i];
                    top_idx[j] = i;
                    break;
                }
            }
        }
        for (int j = 0; j < 5; j++) {
            printf("  [%d] logit=%.3f\n", top_idx[j], top_val[j]);
        }
    }
    fflush(stdout);

    /* Generate new tokens one at a time */
    for (int step = 0; step < max_new; step++) {
        model_forward(&model, full_tokens, total, logits);
        int next_tok = sample(logits, VOCAB_SIZE, top_k, temp);
        full_tokens[total++] = next_tok;

        printf("\r  Step %d/%d  token=%d", step + 1, max_new, next_tok);
        fflush(stdout);

        if (next_tok == 2) break; /* </s> */
        if (total >= MAX_TOKENS * 2 - 1) break;
    }
    printf("\n");

    /* Decode output using vocabulary */
    printf("\n--- Decoded output ---\n");
    for (int i = n_prompt; i < total; i++) {
        int tid = full_tokens[i];
        if (tid >= 0 && tid < model.vocab_size && model.vocab[tid]) {
            /* Skip BOS/EOS/special tokens */
            if (tid != 1 && tid != 2) {
                const char *s = model.vocab[tid];
                /* Replace SentencePiece space marker with actual space */
                for (int j = 0; s[j]; j++) {
                    if (s[j] == '\xe2' && s[j+1] == '\x96' && s[j+2] == '\x81') {
                        putchar(' ');
                        j += 2;
                    } else {
                        putchar(s[j]);
                    }
                }
            }
        } else {
            printf("[tok:%d]", tid);
        }
    }
    printf("\n--- End ---\n");

    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("Generated %d new tokens in %.2fs (%.1f tok/s)\n",
           total - n_prompt, elapsed, (total - n_prompt) / elapsed);

    model_free(&model);
    return 0;
}
