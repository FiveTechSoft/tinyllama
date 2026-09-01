/* ad_gpu.cu - GPU-accelerated training for the adaptive model
 *
 * Forward pass: cuBLAS batched GEMM (all T positions simultaneously per layer).
 * Backward: head-only initially (matches CPU behavior), full backward next.
 * Adam optimizer on GPU.
 *
 * Compile:
 *   nvcc -O3 -arch=sm_86 -c ad_gpu.cu -o ad_gpu.o
 *   nvcc -O3 -arch=sm_86 train.cu ad_gpu.o ad_muon.cu -lcublas -lm
 */
#include "ad_gpu.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CERR(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__,__LINE__, \
                cudaGetErrorString(e)); return -1; \
    } \
} while(0)
#define CBERR(call) do { \
    cublasStatus_t s = (call); \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS %s:%d: %d\n",__FILE__,__LINE__,(int)s); \
        return -1; \
    } \
} while(0)

/* ---- GPU state ---- */
static cublasHandle_t gblas;
static float *d_w, *d_g, *d_m, *d_v;
static float *d_act;
static int   *d_tok;
static AdConfig  gc;
static AdLayout  gl;
static size_t    gnf;
static int       gready = 0, gstep = 0;

/* activation workspace offsets (floats from d_act start) */
static size_t OX,OL1,OQ,OA,OL2,OF1,ODS,OLF,OLG,ODG;
static size_t ACTSZ;

static void calc_layout(int T) {
    int d=gc.dim,h=gc.hidden,V=gc.vocab,L=gc.n_layers;
    size_t o=0;
    OX   =o; o+=(size_t)(L+1)*T*d;
    OL1  =o; o+=(size_t)L*T*d;
    OQ   =o; o+=(size_t)L*T*3*d;
    OA   =o; o+=(size_t)L*T*d;
    OL2  =o; o+=(size_t)L*T*d;
    OF1  =o; o+=(size_t)L*T*h;
    ODS  =o; o+=(size_t)T*d;          /* scratch: proj/ffn2 output */
    OLF  =o; o+=(size_t)T*d;
    OLG  =o; o+=(size_t)T*V;
    ODG  =o; o+=(size_t)T*V;
    ACTSZ=o;
}

/* ==================== KERNELS ==================== */

__global__ void k_zero(float *p,size_t n){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)p[i]=0;}

__global__ void k_embed(float *x,const float *emb,const float *pos,
                        const int *tok,int T,int dim,int V){
    int t=blockIdx.x;if(t>=T)return;
    int id=tok[t]%V;
    for(int i=threadIdx.x;i<dim;i+=blockDim.x)
        x[(size_t)t*dim+i]=emb[(size_t)id*dim+i]+pos[(size_t)t*dim+i];
}

__global__ void k_ln_fwd(float *out,const float *x,const float *g,
                         const float *b,int T,int dim){
    extern __shared__ float sm[];
    int t=blockIdx.x;if(t>=T)return;
    const float *xt=x+(size_t)t*dim;float *s=sm;
    float su=0;for(int i=threadIdx.x;i<dim;i+=blockDim.x)su+=xt[i];
    s[threadIdx.x]=su;__syncthreads();
    for(int q=blockDim.x>>1;q>0;q>>=1){if(threadIdx.x<q)s[threadIdx.x]+=s[threadIdx.x+q];__syncthreads();}
    float mean=s[0]/dim;
    float v=0;for(int i=threadIdx.x;i<dim;i+=blockDim.x){float d=xt[i]-mean;v+=d*d;}
    s[threadIdx.x]=v;__syncthreads();
    for(int q=blockDim.x>>1;q>0;q>>=1){if(threadIdx.x<q)s[threadIdx.x]+=s[threadIdx.x+q];__syncthreads();}
    float iv=1.f/sqrtf(s[0]/dim+1e-5f);
    float *ot=out+(size_t)t*dim;
    for(int i=threadIdx.x;i<dim;i+=blockDim.x)ot[i]=(xt[i]-mean)*iv*g[i]+b[i];
}

__global__ void k_bias(float *out,const float *x,const float *bias,int T,int dim){
    int t=blockIdx.x;if(t>=T)return;
    for(int i=threadIdx.x;i<dim;i+=blockDim.x)
        out[(size_t)t*dim+i]=x[(size_t)t*dim+i]+bias[i];
}

__global__ void k_res(float *out,const float *a,const float *b,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)out[i]=a[i]+b[i];
}
__global__ void k_res_add(float *a,const float *b,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)a[i]+=b[i];
}

__global__ void k_gelu(float *x,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;float v=x[i];
    x[i]=.5f*v*(1.f+tanhf(.7978845608f*(v+.044715f*v*v*v)));
}

__device__ void d_softmax(float *sc,int n){
    float mx=sc[0];for(int i=1;i<n;i++)if(sc[i]>mx)mx=sc[i];
    float s=0;for(int i=0;i<n;i++){sc[i]=expf(sc[i]-mx);s+=sc[i];}
    for(int i=0;i<n;i++)sc[i]/=s;
}

__global__ void k_attn(float *ao,const float *qkv,int T,int dim,int nh,int hd){
    int t=blockIdx.x,h=blockIdx.y;
    if(t>=T||h>=nh)return;
    extern __shared__ float sc[];
    int hid=nh*hd;float scale=1.f/sqrtf((float)hd);
    const float *Q=qkv+(size_t)t*3*hid;
    for(int s=threadIdx.x;s<=t;s+=blockDim.x){
        float dp=0;
        const float *kh=qkv+(size_t)s*3*hid+hid+(size_t)h*hd;
        for(int i=0;i<hd;i++)dp+=Q[(size_t)h*hd+i]*kh[i];
        sc[s]=dp*scale;
    }
    __syncthreads();d_softmax(sc,t+1);__syncthreads();
    float *oh=ao+(size_t)t*dim+(size_t)h*hd;
    for(int i=threadIdx.x;i<hd;i+=blockDim.x){
        float s=0;
        for(int ss=0;ss<=t;ss++)
            s+=sc[ss]*qkv[(size_t)ss*3*hid+2*hid+(size_t)h*hd+i];
        oh[i]=s;
    }
}

/* dlogits = (softmax - onehot) / (T-1) */
__global__ void k_dlog(float *dlg,const float *lg,const int *tgt,int T,int V){
    int t=blockIdx.x;if(t>=T)return;
    float mx=lg[(size_t)t*V];
    for(int i=1;i<V;i++){float v=lg[(size_t)t*V+i];if(v>mx)mx=v;}
    extern __shared__ float rs[];
    float su=0;for(int i=threadIdx.x;i<V;i+=blockDim.x)su+=expf(lg[(size_t)t*V+i]-mx);
    rs[threadIdx.x]=su;__syncthreads();
    for(int d=blockDim.x>>1;d>0;d>>=1){if(threadIdx.x<d)rs[threadIdx.x]+=rs[threadIdx.x+d];__syncthreads();}
    su=rs[0];float inv=1.f/(float)(T-1);
    int ti=tgt[t+1];
    for(int i=threadIdx.x;i<V;i+=blockDim.x){
        float pi=expf(lg[(size_t)t*V+i]-mx)/su;
        dlg[(size_t)t*V+i]=(pi-(i==ti?1.f:0.f))*inv;
    }
}

/* bias grad: dbg[i] += sum_t dout[t,i] */
__global__ void k_bgrad(float *dbg,const float *dout,int T,int dim){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=dim)return;float s=0;
    for(int t=0;t<T;t++)s+=dout[(size_t)t*dim+i];
    atomicAdd(&dbg[i],s);
}

__global__ void k_adam(float *w,float *g,float *m,float *v,
                       size_t n,float lr,float bc1,float bc2,float eps){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    float gi=g[i];
    m[i]=.9f*m[i]+.1f*gi;v[i]=.999f*v[i]+.001f*gi*gi;
    w[i]-=lr*(m[i]/bc1)/(sqrtf(v[i]/bc2)+1e-8f);
    g[i]=0.f;
}

/* gradient scale (batch norm + clip) */
__global__ void k_scale(float *g,float s,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)g[i]*=s;
}

/* ==================== cuBLAS WRAPPERS ====================
 * Row-major: OUT[T,R] = W[R,C] × X[T,C]^T
 * Cublas column-major:
 *   forward:  cublasSgemm(NT, R, T, C, a, W, C, X, C, b, OUT, R)
 *   dW:       cublasSgemm(NT, C, R, T, a, X, C, dOut, R, b=1, dW, C)  (accumulate)
 *   dX:       cublasSgemm(NN, C, T, R, a, W, C, dOut, R, b, dX, C)
 */
static int gm_fwd(float *o,const float *W,const float *X,int R,int C,int T){
    float a=1,b=0;
    CBERR(cublasSgemm(gblas,CUBLAS_OP_T,CUBLAS_OP_N,R,T,C,&a,W,C,X,C,&b,o,R));
    return 0;
}
static int gm_dW(float *dW,const float *X,const float *dO,int R,int C,int T){
    float a=1,b=1;
    CBERR(cublasSgemm(gblas,CUBLAS_OP_N,CUBLAS_OP_T,C,R,T,&a,X,C,dO,R,&b,dW,C));
    return 0;
}
static int gm_dX(float *dX,const float *W,const float *dO,int R,int C,int T){
    float a=1,b=0;
    CBERR(cublasSgemm(gblas,CUBLAS_OP_N,CUBLAS_OP_N,C,T,R,&a,W,C,dO,R,&b,dX,C));
    return 0;
}

/* ==================== FORWARD ==================== */
static int gpu_forward(const int *tokens,int T){
    int d=gc.dim,hd=gc.head_dim,nh=gc.n_heads,L=gc.n_layers;
    int hid=gc.hidden,V=gc.vocab;
    size_t D=(size_t)d,H=(size_t)hid;

    CERR(cudaMemcpy(d_tok,tokens,T*sizeof(int),cudaMemcpyHostToDevice));
    k_zero<<<(gnf+255)/256,256>>>(d_g,gnf);
    cudaDeviceSynchronize();

    /* embedding */
    k_embed<<<T,256>>>(d_act+OX,d_w+gl.tok_emb,d_w+gl.pos_emb,d_tok,T,d,V);
    cudaDeviceSynchronize();

    /* per-layer forward */
    for(int l=0;l<L;l++){
        size_t lb=gl.layers+(size_t)l*gl.per_layer;
        const float *ln1g=d_w+lb,*ln1b=ln1g+D;
        const float *Wq=ln1b+D,*bq=Wq+D*3*D;
        const float *Wp=bq+3*D,*bp=Wp+D*D;
        const float *ln2g=bp+D,*ln2b=ln2g+D;
        const float *W1=ln2b+D,*b1=W1+D*H;
        const float *W2=b1+H,*b2=W2+H*D;

        size_t ox=(size_t)l*T*d, ox2=(size_t)(l+1)*T*d;
        size_t oL1=(size_t)l*T*d, oQ=(size_t)l*T*3*d;
        size_t oA=(size_t)l*T*d, oL2=(size_t)l*T*d;
        size_t oF=(size_t)l*T*hid;

        /* LN1 */
        k_ln_fwd<<<T,256,256*sizeof(float)>>>(
            d_act+OL1+oL1, d_act+OX+ox, ln1g,ln1b,T,d);
        cudaDeviceSynchronize();
        /* QKV */
        gm_fwd(d_act+OQ+oQ, Wq, d_act+OL1+oL1, 3*d,d,T);
        k_bias<<<T,256>>>(d_act+OQ+oQ, d_act+OQ+oQ, bq, T,3*d);
        cudaDeviceSynchronize();
        /* attention */
        {dim3 g(T,nh);
         k_attn<<<g,min(hd,256),T*sizeof(float)>>>(
            d_act+OA+oA, d_act+OQ+oQ, T,d,nh,hd);}
        cudaDeviceSynchronize();
        /* proj -> ODS, add bp, residual */
        gm_fwd(d_act+ODS, Wp, d_act+OA+oA, d,d,T);
        k_bias<<<T,256>>>(d_act+ODS, d_act+ODS, bp, T,d);
        k_res<<<(T*d+255)/256,256>>>(d_act+OX+ox2, d_act+OX+ox, d_act+ODS, T*d);
        cudaDeviceSynchronize();
        /* LN2 */
        k_ln_fwd<<<T,256,256*sizeof(float)>>>(
            d_act+OL2+oL2, d_act+OX+ox2, ln2g,ln2b,T,d);
        cudaDeviceSynchronize();
        /* FFN W1 + bias + GELU */
        gm_fwd(d_act+OF1+oF, W1, d_act+OL2+oL2, hid,d,T);
        k_bias<<<T,256>>>(d_act+OF1+oF, d_act+OF1+oF, b1, T,hid);
        cudaDeviceSynchronize();
        k_gelu<<<(T*hid+255)/256,256>>>(d_act+OF1+oF, T*hid);
        cudaDeviceSynchronize();
        /* FFN W2 -> ODS, add b2, residual */
        gm_fwd(d_act+ODS, W2, d_act+OF1+oF, d,hid,T);
        k_bias<<<T,256>>>(d_act+ODS, d_act+ODS, b2, T,d);
        k_res_add<<<(T*d+255)/256,256>>>(d_act+OX+ox2, d_act+ODS, T*d);
        cudaDeviceSynchronize();
    }

    /* final LN + head */
    k_ln_fwd<<<T,256,256*sizeof(float)>>>(
        d_act+OLF, d_act+OX+(size_t)L*T*d, d_w+gl.lnf_g, d_w+gl.lnf_b, T,d);
    cudaDeviceSynchronize();
    gm_fwd(d_act+OLG, d_w+gl.w_head, d_act+OLF, V,d,T);
    k_bias<<<T,256>>>(d_act+OLG, d_act+OLG, d_w+gl.b_head, T,V);
    cudaDeviceSynchronize();

    return 0;
}

/* ==================== HEAD BACKWARD (matches CPU) ==================== */
static int gpu_head_bwd(const int *tokens,int T){
    int d=gc.dim,V=gc.vocab;
    size_t D=(size_t)d;

    /* compute dlogits on GPU */
    k_dlog<<<T,min(V,1024)>>>(d_act+ODG, d_act+OLG, d_tok, T,V);
    cudaDeviceSynchronize();

    /* dWhead += dlogits^T @ lf */
    gm_dW(d_g+gl.w_head, d_act+OLF, d_act+ODG, V,d,T);

    /* dbhead += sum_t dlogits[t] */
    k_bgrad<<<(V+255)/256,256>>>(d_g+gl.b_head, d_act+ODG, T,V);
    cudaDeviceSynchronize();

    return 0;
}

/* ==================== LOSS ON CPU ==================== */
static float cpu_loss(const int *tokens,int T){
    int V=gc.vocab;
    float *h_lg=(float*)malloc((size_t)T*V*sizeof(float));
    CERR(cudaMemcpy(h_lg,d_act+OLG,(size_t)T*V*sizeof(float),
                    cudaMemcpyDeviceToHost));
    double ce=0;
    for(int t=0;t<T-1;t++){
        float mx=h_lg[(size_t)t*V];
        for(int i=1;i<V;i++)if(h_lg[(size_t)t*V+i]>mx)mx=h_lg[(size_t)t*V+i];
        double s=0;
        for(int i=0;i<V;i++)s+=exp((double)h_lg[(size_t)t*V+i]-mx);
        int tgt=tokens[t+1]%V;
        ce+=-((double)h_lg[(size_t)t*V+tgt]-mx-log(s));
    }
    free(h_lg);
    return(float)(ce/(T-1));
}

/* ==================== PUBLIC API ==================== */

int ad_gpu_init(AdModel *m){
    if(gready)return 0;
    CERR(cudaSetDevice(0));
    CBERR(cublasCreate(&gblas));
    CBERR(cublasSetMathMode(gblas,CUBLAS_DEFAULT_MATH));

    gc=m->cfg; gl=m->lay; gnf=ad_total_floats(&m->cfg);

    CERR(cudaMalloc(&d_w,gnf*sizeof(float)));
    CERR(cudaMemcpy(d_w,m->w,gnf*sizeof(float),cudaMemcpyHostToDevice));
    CERR(cudaMalloc(&d_g,gnf*sizeof(float)));
    CERR(cudaMalloc(&d_m,gnf*sizeof(float)));
    CERR(cudaMalloc(&d_v,gnf*sizeof(float)));
    CERR(cudaMemset(d_g,0,gnf*sizeof(float)));
    CERR(cudaMemset(d_m,0,gnf*sizeof(float)));
    CERR(cudaMemset(d_v,0,gnf*sizeof(float)));

    calc_layout(gc.max_seq);
    CERR(cudaMalloc(&d_act,ACTSZ*sizeof(float)));
    CERR(cudaMalloc(&d_tok,gc.max_seq*sizeof(int)));

    gready=1; gstep=0;
    printf("GPU: %.1fMB weights, %.1fMB act\n",
           (double)gnf*4/1e6,(double)ACTSZ*4/1e6);
    return 0;
}

float ad_gpu_fwd_bwd(AdModel *m,const int *tokens,int T,int do_bwd){
    (void)m;
    calc_layout(T);
    gpu_forward(tokens,T);
    float loss=cpu_loss(tokens,T);
    if(do_bwd) gpu_head_bwd(tokens,T);
    return loss;
}

void ad_gpu_adam_step(float lr,float beta1,float beta2,float eps,float wd){
    (void)wd;
    gstep++;
    float bc1=1.f-powf(beta1,(float)gstep);
    float bc2=1.f-powf(beta2,(float)gstep);
    /* gradient scale: 1/batch (caller divides by batch before calling) */
    k_scale<<<(gnf+255)/256,256>>>(d_g,gnf,1.f);
    /* gradient clip: L2 norm > 1.0 */
    /* (simplified: skip clipping for now, add cuBLAS nrm2 later) */
    k_adam<<<(gnf+255)/256,256>>>(d_w,d_g,d_m,d_v,gnf,lr,bc1,bc2,eps);
    cudaDeviceSynchronize();
}

void ad_gpu_download(AdModel *m){
    if(!gready)return;
    cudaError_t e=cudaMemcpy(m->w,d_w,gnf*sizeof(float),cudaMemcpyDeviceToHost);
    if(e!=cudaSuccess)fprintf(stderr,"CUDA download: %s\n",cudaGetErrorString(e));
}

void ad_gpu_free(void){
    if(!gready)return;
    cudaFree(d_w);cudaFree(d_g);cudaFree(d_m);cudaFree(d_v);
    cudaFree(d_act);cudaFree(d_tok);
    cublasDestroy(gblas);
    d_w=d_g=d_m=d_v=d_act=NULL;d_tok=NULL;gready=0;
}

int ad_gpu_ready(void){return gready;}
