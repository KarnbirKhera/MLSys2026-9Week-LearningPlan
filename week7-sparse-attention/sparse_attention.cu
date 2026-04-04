#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <stdint.h>

#define NUM_HEADS   64
#define HEAD_DIM    128
#define PAGE_SIZE   64
#define TOP_K       2048
#define TILE_K      64

#define PAGE_DATA_BYTES  (PAGE_SIZE * HEAD_DIM)
#define PAGE_SCALE_BYTES (PAGE_SIZE * (int)sizeof(float))
#define PAGE_STRIDE      (PAGE_DATA_BYTES + PAGE_SCALE_BYTES)


using bfloat16 = __nv_bfloat16;


// Q is [batch_size, NUM_HEADS, HEAD_DIM]
__global__ void sparse_attention_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const uint8_t*       __restrict__ K_cache,
    const uint8_t*       __restrict__ V_cache,
    const int*           __restrict__ page_table,
    const int*           __restrict__ topk_indices,
    __nv_bfloat16*       __restrict__ out,
    float*               __restrict__ lse_out,
    int   max_pages,
    float qk_scale
) {
/**************************** Initialize row pointers & Shared memory***************************************************** */
    int batch = blockIdx.x;
    int head  = blockIdx.y;
    int tid   = threadIdx.x;

    __shared__ float smem_q[HEAD_DIM];
    __shared__ float smem_kv[TILE_K][HEAD_DIM + 1];
    __shared__ float smem_dot[TILE_K];
    __shared__ float smem_reduce[HEAD_DIM];

/**************************** Load our Q matrix into smem_q***************************************************** */
    //      BFloat16 to Float32
    smem_q[tid] = __bfloat162float(
        //
        Q[batch * NUM_HEADS * HEAD_DIM +  // [Current Block] * How Many Heads * Size of Head
            head * HEAD_DIM +             // [Current Head]  * Size of Head
            tid                           // [Current thread] * 1
        ]
    );
    __syncthreads();

/**************************** STATE VARIABLES Softmax intialization values***************************************************** */

    float running_max   = -FLT_MAX;
    float running_denom = 0.0f;
    float out_acc       = 0.0f;

/**************************** LOAD our current page ***************************************************** */
    // How many tiles iterations to cover TOP_K
    int num_tiles = (TOP_K + TILE_K - 1) / TILE_K;
    for (int tile = 0; tile < num_tiles; tile++) {

        // What is the start of our current tile
        //            [Current Tile] * [Size of Tile]
        //              [32] * [64] = 2048, covers all of topk 
        int tile_base = tile * TILE_K;
        
        // Prevents processing going out of bounds
        //                  [64]   [2048 - 2048] -> [2048 - 1984] = [64]
        int tile_len  = min(TILE_K, TOP_K - tile_base);

        //Iterate throughout our tile which at a maximum is 64, or 0 (With the current competition values assumed since 2048 % 64 = 0)
        for (int tok = 0; tok < tile_len; tok++) {

            // Where are we within our current tile
            //      [Current Batch] * [Size of Batch] + [Current Tile & Size] + [Offset within tile]
            int global_tok = topk_indices[batch * TOP_K + tile_base + tok]; //topk_indices tensor is  [batch_size,tile_K]

            // Get our row
            int log_page   = global_tok / PAGE_SIZE;

            // Get our column
            int page_off   = global_tok % PAGE_SIZE;

            // Get our current page from our flattened 2D dimension
            //       [Current Batch] * [Size of Batch] + [Our current Page]
            int phys_page  = page_table[batch * max_pages + log_page];

            // Stride between pages in bytes
            //                     [K_cache starting point] + [Size of page in bytes] * [How big the page is]
            const uint8_t* k_page = K_cache + (size_t)phys_page * PAGE_STRIDE;

/**************************** LOAD our k_scale ***************************************************** */

            // Tell compiler we are working with float32
            float k_scale = *reinterpret_cast<const float*>(
                // Where we are in our current page in bytes
                // [Current Page] + [Size of Page in Bytes] + [Page column * size in bytes]
                k_page + PAGE_DATA_BYTES + page_off * sizeof(float)
            );

/**************************** LOAD our k_page value & SCALE said value ***************************************************** */
            // Our FP8
            __nv_fp8_e4m3 fp8_k;
            
            // Copy our K_page value into our fp8_k
            //       To                From
            //             [Current Page] + [Page column] * [Size of column (row)] + [Current location in row]      
            memcpy(&fp8_k, k_page + page_off * HEAD_DIM + tid, 1);
            smem_kv[tok][tid] = __half2float((__half)fp8_k) * k_scale;
        }

        // Sync since we are moving from data moving to data processing
        __syncthreads();

/**************************** COMPUTE, DOT, REDUCE our stored Q values, and K values***************************************************** */

        //Iterate over our tile
        for (int tok = 0; tok < tile_len; tok++) {
            // Tile reduction = Q[Current Value] * [Current Token][Current Head]
            smem_reduce[tid] = smem_q[tid] * smem_kv[tok][tid];
            __syncthreads();

            //Iterate over head dim, reduction loop
            for (int s = HEAD_DIM / 2; s > 0; s >>= 1) {
                //Ensure current thread is within bounds of our current range of reduction
                if (tid < s)
                    //Sum our selected value with our current
                    smem_reduce[tid] += smem_reduce[tid + s];
                __syncthreads();
            }
            //Last thread stores into our smem
            if (tid == 0)
                smem_dot[tok] = smem_reduce[0] * qk_scale;
            __syncthreads();
        }

/**************************** COMPUTE our softmax runningSum,  ***************************************************** */

        //Also this seemed very in-efficient, serializing a single thread across our entire tile seems rather inefficient. We can likely use some sort of reduction.
        // Maybe a tree reductionn across the tile? We would have to take into account our threads would have to cover an X sized tile (changing tile sizes)
        float tile_max = -FLT_MAX;
        //First thread in block
        if (tid == 0) {
            //Iterate through our tile
            for (int tok = 0; tok < tile_len; tok++)
                //Compare current value to local tile max
                tile_max = fmaxf(tile_max, smem_dot[tok]);
            // Re-use our reduce SMEM
            smem_reduce[0] = tile_max;
        }
        
        // AHHH I SEE. I was confused because if we're only using the first thread within our block, I know that our first warp (the one that contains thread 0) would diverge. So thread 0 - 31 are dealt with.
        // And I assumed that for threads 32 and beyond, they would also be considered "offline" because they all are not tid == 0, hence the entire warp must of been diverged. But the thing is, the entire WARP
        // does not pass the tid=0 meaning there is no divergence, and the warp would actually still be active. This is why we need our syncthreads here
        __syncthreads();

        //Set our current tile_max to our local tile_max
        tile_max = smem_reduce[0];

        // Compare our current tile_max to our state variable running max
        float new_max = fmaxf(running_max, tile_max);

        //Calculate the needed A * B, respective B term for rescaling where the entire formula for rescaling is A * B + C
        float rescale = expf(running_max - new_max);

        // Our local tiles denominator
        float tile_denom = 0.0f;

        //Only the first thread within our block
        if (tid == 0) {
            for (int tok = 0; tok < tile_len; tok++)
                //Ensure numerical stability by subtracting our current number by our maximum
                tile_denom += expf(smem_dot[tok] - new_max);
            //Store our tile_denom within smem
            smem_reduce[0] = tile_denom;
        }

        // Same reasoning as prior thread barrier
        __syncthreads();
        // Shared memory -> Local register
        tile_denom = smem_reduce[0];

        // Rescale our output A * B, still requires + C term later on
        out_acc       *= rescale;

        //We rescale our running denom, using our A * B + C formula, (old Dimension) * (old to new Dimension) + (offset within new dimension)
        running_denom  = running_denom * rescale + tile_denom;

        //Set our local tile new max to our state variable running_max
        running_max    = new_max;

        // Iterate over our tile
        for (int tok = 0; tok < tile_len; tok++) {
            //                          [Current Batch] * [Size of Batch] + [Current Tile & Size] + [Offset within tile]
            int global_tok = topk_indices[batch * TOP_K + tile_base + tok]; //topk_indices tensor is  [batch_size,tile_K, page_row*page_col]

            //Our current page (row)
            int log_page   = global_tok / PAGE_SIZE;
            
            // Our current column within page
            int page_off   = global_tok % PAGE_SIZE;

            // Our current physical address of our page
            //                      [Current batch] * [Pages per Batch] + [Our current page]
            int phys_page  = page_table[batch * max_pages + log_page];

            // Start of our v_page within v_cache taking into account the current physical page locatoin
            //                   [Start of our V_cache address] + [Size of our page in bytes] * [How many pages to reach next row]
            const uint8_t* v_page = V_cache + (size_t)phys_page * PAGE_STRIDE;

            // Tell compiler we will be using FP32
            float v_scale = *reinterpret_cast<const float*>(

                // Our current page 
                // Current V_Page location + [Size of page in memory] * [Size of the element within the page (FP32)]
                v_page + PAGE_DATA_BYTES + page_off * sizeof(float)
            );

            // FP8 
            __nv_fp8_e4m3 fp8_v;

            // Copy our V_page value to our local variable
            //      To                  From         Size in Bytes (8 Bits -> 1 Byte)           
            memcpy(&fp8_v, v_page + page_off * HEAD_DIM + tid, 1);

            //Scale our FP8 V value into FP32, reuse our smem_kv cache for values (rather than k which we did before)
            smem_kv[tok][tid] = __half2float((__half)fp8_v) * v_scale;
        }

        //Ensure our smem_kv is fully populated before using the data
        __syncthreads();

        //Iterate through out our tile
        for (int tok = 0; tok < tile_len; tok++) {
            // Ensure numerical stability by subtracting by running max for our Q * K dot product (smem_dot)
            float w = expf(smem_dot[tok] - running_max);
        
            // Our + C term for our output_acc which we referenced needing earlier.
            // We multiply our Q * K dot product (w) by our V value to give it proper weight
            out_acc += w * smem_kv[tok][tid];
        }
        __syncthreads();
    }

    // Our softmax 0-1
    out_acc /= running_denom;

    // Out tensor is [batch_size, NUM_HEADS * HEAD_DIM]  where NUM_HEADS * HEAD_DIM are flattened into a single dimension and were originally two dimensions... 
    //Hmm this doesnt match our addressing style for a paged dimension (/ %) but rather matches our multi-stride arithmetic leaf in  our framework
    //
    // Our multi-stride formula is just Index * Stride recursively called. So this formula essentially Index * Stride just presented differently.
    // [Current Batch] * [Num Of Heads in Batch] * [Num of Elements in Head] + [Offset within Head (Batch)] * [Size of each Head] + [Where we are within the head] 
    // It is preferred to be presented this way as it allows us to easliy read our tensor layout of [batch_size, NUM_HEADS, HEAD_DIM]
    // [Z Index * (Size of Y * Size of X) + (Y Index * Size of X) + X Index]
    out[batch * NUM_HEADS * HEAD_DIM + head * HEAD_DIM + tid] =
        //Convert our FP32 into an BFLOAT16
        __float2bfloat16(out_acc);

    // 
    if (tid == 0)
        // Store the our Log Sum Exponential running sum of this row so we can compute them together with our other rows LSE values.
        // During back propagation, rather than storing all of attention probabilties values (Q * K^T/sqrt(d)), we can simply store our smoothed (LSE) runningSum
        // And since we know the numbers that contributed to our runningSum havent changed, we can use our smoothed running sum for when we 
        // calculate our softmax during back propagation. 

        // We do logf to inverse our e^running_denominator which allows us to simply add our maximum to our running denominator
        lse_out[batch * NUM_HEADS + head] = running_max + logf(running_denom);

        //If we do max(x,a) we have a piecewise function where a dominates until x finally catches up, resulting in a upward curve (Piece wise function).
        // We use LSE to this sharp corner into a smooth curve that we can properly find the derivative of. Where if we use the max function (sharp corner) we have
        // the following derivative
        // (x < a) = 0
        // (x > a) = 1
        // (x = a) = undefined
        // This proves to be a challenge when during back propagation, if our derivative is undefined, we require a complex workaround which could have been solved at the root.
        //
        // We use LSE to smoothen this sharp corner, allowing each individual point to have a derivative (rate of change) we can calculate, allowing for our back propagation
        // to properly calculate the gradient of our loss function (to properly update each individual weight in respect to their partial derivative contribution to the loss function)
}
















static void sparse_attention_cpu(
    const float*   Q,
    const uint8_t* K_cache,
    const uint8_t* V_cache,
    const int*     page_table,
    const int*     topk_indices,
    float*         out,
    float*         lse_out,
    int batch_size, int max_pages, float qk_scale
) {
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < NUM_HEADS; h++) {
            const float* q = Q + b * NUM_HEADS * HEAD_DIM + h * HEAD_DIM;

            float* scores = (float*)malloc(TOP_K * sizeof(float));
            float  row_max = -FLT_MAX;

            for (int ki = 0; ki < TOP_K; ki++) {
                int tok      = topk_indices[b * TOP_K + ki];
                int log_page = tok / PAGE_SIZE;
                int page_off = tok % PAGE_SIZE;
                int phys     = page_table[b * max_pages + log_page];

                const uint8_t* kp = K_cache + (size_t)phys * PAGE_STRIDE;
                float k_sc = *reinterpret_cast<const float*>(kp + PAGE_DATA_BYTES + page_off * sizeof(float));

                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) {
                    __nv_fp8_e4m3 fp8_k;
                    memcpy(&fp8_k, kp + page_off * HEAD_DIM + d, 1);
                    float kd = __half2float((__half)fp8_k) * k_sc;
                    dot += q[d] * kd;
                }
                scores[ki] = dot * qk_scale;
                row_max = fmaxf(row_max, scores[ki]);
            }

            float denom = 0.0f;
            for (int ki = 0; ki < TOP_K; ki++)
                denom += expf(scores[ki] - row_max);

            float* out_h = out + b * NUM_HEADS * HEAD_DIM + h * HEAD_DIM;
            memset(out_h, 0, HEAD_DIM * sizeof(float));

            for (int ki = 0; ki < TOP_K; ki++) {
                float w = expf(scores[ki] - row_max) / denom;

                int tok      = topk_indices[b * TOP_K + ki];
                int log_page = tok / PAGE_SIZE;
                int page_off = tok % PAGE_SIZE;
                int phys     = page_table[b * max_pages + log_page];

                const uint8_t* vp = V_cache + (size_t)phys * PAGE_STRIDE;
                float v_sc = *reinterpret_cast<const float*>(vp + PAGE_DATA_BYTES + page_off * sizeof(float));

                for (int d = 0; d < HEAD_DIM; d++) {
                    __nv_fp8_e4m3 fp8_v;
                    memcpy(&fp8_v, vp + page_off * HEAD_DIM + d, 1);
                    float vd = __half2float((__half)fp8_v) * v_sc;
                    out_h[d] += w * vd;
                }
            }

            lse_out[b * NUM_HEADS + h] = row_max + logf(denom);
            free(scores);
        }
    }
}


static void fill_rand(float* buf, int n) {
    for (int i = 0; i < n; i++) buf[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
}

static float max_abs_error(const float* a, const float* b, int n) {
    float e = 0.f;
    for (int i = 0; i < n; i++) e = fmaxf(e, fabsf(a[i] - b[i]));
    return e;
}

static uint8_t float_to_fp8_byte(float v) {
    __nv_fp8_e4m3 fp8 = (__nv_fp8_e4m3)(__half)__float2half(v);
    uint8_t raw;
    memcpy(&raw, &fp8, 1);
    return raw;
}

static uint8_t* build_kv_cache(int num_phys_pages) {
    size_t total = (size_t)num_phys_pages * PAGE_STRIDE;
    uint8_t* buf = (uint8_t*)malloc(total);
    for (int pg = 0; pg < num_phys_pages; pg++) {
        uint8_t* page = buf + (size_t)pg * PAGE_STRIDE;
        for (int i = 0; i < PAGE_SIZE * HEAD_DIM; i++) {
            float v = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
            page[i] = float_to_fp8_byte(v);
        }
        float* scales = reinterpret_cast<float*>(page + PAGE_DATA_BYTES);
        for (int t = 0; t < PAGE_SIZE; t++) scales[t] = 1.0f;
    }
    return buf;
}


static void test_sparse_attention_correctness() {
    printf("\n=== test_sparse_attention_correctness ===\n");

    const int batch_size = 2;
    const int num_pages  = (TOP_K * 2) / PAGE_SIZE;

    int q_elems = batch_size * NUM_HEADS * HEAD_DIM;
    float* h_Q_fp32 = (float*)malloc(q_elems * sizeof(float));
    fill_rand(h_Q_fp32, q_elems);

    __nv_bfloat16* h_Q_bf16 = (__nv_bfloat16*)malloc(q_elems * sizeof(__nv_bfloat16));
    for (int i = 0; i < q_elems; i++)
        h_Q_bf16[i] = __float2bfloat16(h_Q_fp32[i]);

    uint8_t* h_K = build_kv_cache(num_pages);
    uint8_t* h_V = build_kv_cache(num_pages);

    int* h_pt = (int*)malloc(batch_size * num_pages * sizeof(int));
    for (int b = 0; b < batch_size; b++)
        for (int p = 0; p < num_pages; p++)
            h_pt[b * num_pages + p] = p;

    int* h_topk = (int*)malloc(batch_size * TOP_K * sizeof(int));
    for (int b = 0; b < batch_size; b++)
        for (int ki = 0; ki < TOP_K; ki++)
            h_topk[b * TOP_K + ki] = ki;

    float* h_out_cpu  = (float*)malloc(batch_size * NUM_HEADS * HEAD_DIM * sizeof(float));
    float* h_lse_cpu  = (float*)malloc(batch_size * NUM_HEADS * sizeof(float));
    float* h_out_gpu  = (float*)malloc(batch_size * NUM_HEADS * HEAD_DIM * sizeof(float));
    float* h_lse_gpu  = (float*)malloc(batch_size * NUM_HEADS * sizeof(float));
    __nv_bfloat16* h_out_bf16 = (__nv_bfloat16*)malloc(batch_size * NUM_HEADS * HEAD_DIM * sizeof(__nv_bfloat16));

    float qk_scale = 1.0f / sqrtf((float)HEAD_DIM);

    sparse_attention_cpu(
        h_Q_fp32, h_K, h_V, h_pt, h_topk,
        h_out_cpu, h_lse_cpu,
        batch_size, num_pages, qk_scale
    );

    __nv_bfloat16 *d_Q, *d_out;
    uint8_t *d_K, *d_V;
    int *d_pt, *d_topk;
    float *d_lse;

    cudaMalloc(&d_Q,    q_elems * sizeof(__nv_bfloat16));
    cudaMalloc(&d_K,    (size_t)num_pages * PAGE_STRIDE);
    cudaMalloc(&d_V,    (size_t)num_pages * PAGE_STRIDE);
    cudaMalloc(&d_pt,   batch_size * num_pages * sizeof(int));
    cudaMalloc(&d_topk, batch_size * TOP_K * sizeof(int));
    cudaMalloc(&d_out,  batch_size * NUM_HEADS * HEAD_DIM * sizeof(__nv_bfloat16));
    cudaMalloc(&d_lse,  batch_size * NUM_HEADS * sizeof(float));

    cudaMemcpy(d_Q,    h_Q_bf16, q_elems * sizeof(__nv_bfloat16),      cudaMemcpyHostToDevice);
    cudaMemcpy(d_K,    h_K,      (size_t)num_pages * PAGE_STRIDE,       cudaMemcpyHostToDevice);
    cudaMemcpy(d_V,    h_V,      (size_t)num_pages * PAGE_STRIDE,       cudaMemcpyHostToDevice);
    cudaMemcpy(d_pt,   h_pt,     batch_size * num_pages * sizeof(int),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_topk, h_topk,   batch_size * TOP_K * sizeof(int),      cudaMemcpyHostToDevice);

    dim3 grid(batch_size, NUM_HEADS);
    dim3 block(HEAD_DIM);

    sparse_attention_kernel<<<grid, block>>>(
        d_Q, d_K, d_V, d_pt, d_topk,
        d_out, d_lse,
        num_pages, qk_scale
    );

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaMemcpy(h_out_bf16, d_out, batch_size * NUM_HEADS * HEAD_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_lse_gpu,  d_lse, batch_size * NUM_HEADS * sizeof(float),                    cudaMemcpyDeviceToHost);

    int out_elems = batch_size * NUM_HEADS * HEAD_DIM;
    for (int i = 0; i < out_elems; i++)
        h_out_gpu[i] = __bfloat162float(h_out_bf16[i]);

    float out_err = max_abs_error(h_out_gpu, h_out_cpu, out_elems);
    float lse_err = max_abs_error(h_lse_gpu, h_lse_cpu, batch_size * NUM_HEADS);

    printf("Output error: max_err = %.2e  %s\n", out_err, out_err < 1e-2f ? "PASS" : "FAIL");
    printf("LSE error:    max_err = %.2e  %s\n", lse_err, lse_err < 1e-3f ? "PASS" : "FAIL");

    free(h_Q_fp32); free(h_Q_bf16); free(h_K); free(h_V);
    free(h_pt); free(h_topk);
    free(h_out_cpu); free(h_lse_cpu); free(h_out_gpu); free(h_lse_gpu); free(h_out_bf16);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_pt);
    cudaFree(d_topk); cudaFree(d_out); cudaFree(d_lse);
}


static void test_paged_scatter() {
    printf("\n=== test_paged_scatter (shuffled page table) ===\n");

    const int batch_size = 1;
    const int num_pages  = TOP_K / PAGE_SIZE;

    int q_elems = batch_size * NUM_HEADS * HEAD_DIM;
    float* h_Q_fp32 = (float*)malloc(q_elems * sizeof(float));
    fill_rand(h_Q_fp32, q_elems);

    __nv_bfloat16* h_Q_bf16 = (__nv_bfloat16*)malloc(q_elems * sizeof(__nv_bfloat16));
    for (int i = 0; i < q_elems; i++)
        h_Q_bf16[i] = __float2bfloat16(h_Q_fp32[i]);

    uint8_t* h_K = build_kv_cache(num_pages);
    uint8_t* h_V = build_kv_cache(num_pages);

    int* h_pt = (int*)malloc(batch_size * num_pages * sizeof(int));
    for (int p = 0; p < num_pages; p++)
        h_pt[p] = num_pages - 1 - p;

    int* h_topk = (int*)malloc(batch_size * TOP_K * sizeof(int));
    for (int ki = 0; ki < TOP_K; ki++)
        h_topk[ki] = ki;

    float* h_out_cpu = (float*)malloc(batch_size * NUM_HEADS * HEAD_DIM * sizeof(float));
    float* h_lse_cpu = (float*)malloc(batch_size * NUM_HEADS * sizeof(float));
    float* h_out_gpu = (float*)malloc(batch_size * NUM_HEADS * HEAD_DIM * sizeof(float));
    float* h_lse_gpu = (float*)malloc(batch_size * NUM_HEADS * sizeof(float));
    __nv_bfloat16* h_out_bf16 = (__nv_bfloat16*)malloc(batch_size * NUM_HEADS * HEAD_DIM * sizeof(__nv_bfloat16));

    float qk_scale = 1.0f / sqrtf((float)HEAD_DIM);

    sparse_attention_cpu(
        h_Q_fp32, h_K, h_V, h_pt, h_topk,
        h_out_cpu, h_lse_cpu,
        batch_size, num_pages, qk_scale
    );

    __nv_bfloat16 *d_Q, *d_out;
    uint8_t *d_K, *d_V;
    int *d_pt, *d_topk;
    float *d_lse;

    cudaMalloc(&d_Q,    q_elems * sizeof(__nv_bfloat16));
    cudaMalloc(&d_K,    (size_t)num_pages * PAGE_STRIDE);
    cudaMalloc(&d_V,    (size_t)num_pages * PAGE_STRIDE);
    cudaMalloc(&d_pt,   batch_size * num_pages * sizeof(int));
    cudaMalloc(&d_topk, batch_size * TOP_K * sizeof(int));
    cudaMalloc(&d_out,  batch_size * NUM_HEADS * HEAD_DIM * sizeof(__nv_bfloat16));
    cudaMalloc(&d_lse,  batch_size * NUM_HEADS * sizeof(float));

    cudaMemcpy(d_Q,    h_Q_bf16, q_elems * sizeof(__nv_bfloat16),      cudaMemcpyHostToDevice);
    cudaMemcpy(d_K,    h_K,      (size_t)num_pages * PAGE_STRIDE,       cudaMemcpyHostToDevice);
    cudaMemcpy(d_V,    h_V,      (size_t)num_pages * PAGE_STRIDE,       cudaMemcpyHostToDevice);
    cudaMemcpy(d_pt,   h_pt,     batch_size * num_pages * sizeof(int),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_topk, h_topk,   batch_size * TOP_K * sizeof(int),      cudaMemcpyHostToDevice);

    dim3 grid(batch_size, NUM_HEADS);
    dim3 block(HEAD_DIM);

    sparse_attention_kernel<<<grid, block>>>(
        d_Q, d_K, d_V, d_pt, d_topk,
        d_out, d_lse,
        num_pages, qk_scale
    );

    cudaDeviceSynchronize();
    cudaMemcpy(h_out_bf16, d_out, batch_size * NUM_HEADS * HEAD_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_lse_gpu,  d_lse, batch_size * NUM_HEADS * sizeof(float),                    cudaMemcpyDeviceToHost);

    int out_elems = batch_size * NUM_HEADS * HEAD_DIM;
    for (int i = 0; i < out_elems; i++)
        h_out_gpu[i] = __bfloat162float(h_out_bf16[i]);

    float out_err = max_abs_error(h_out_gpu, h_out_cpu, out_elems);
    float lse_err = max_abs_error(h_lse_gpu, h_lse_cpu, batch_size * NUM_HEADS);

    printf("Output error: max_err = %.2e  %s\n", out_err, out_err < 1e-2f ? "PASS" : "FAIL");
    printf("LSE error:    max_err = %.2e  %s\n", lse_err, lse_err < 1e-3f ? "PASS" : "FAIL");

    free(h_Q_fp32); free(h_Q_bf16); free(h_K); free(h_V);
    free(h_pt); free(h_topk);
    free(h_out_cpu); free(h_lse_cpu); free(h_out_gpu); free(h_lse_gpu); free(h_out_bf16);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_pt);
    cudaFree(d_topk); cudaFree(d_out); cudaFree(d_lse);
}


int main() {
    srand(42);
    printf("========== Week 7: Sparse Attention Kernel ==========\n");
    test_sparse_attention_correctness();
    test_paged_scatter();
    printf("\n=====================================================\n");
    return 0;
}
