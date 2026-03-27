#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <algorithm>


// Our competition inputs 
#define NUM_HEADS   64
#define HEAD_DIM    128
#define PAGE_SIZE   64
#define TOP_K       2048

#define PAGE_DATA_BYTES   (PAGE_SIZE * HEAD_DIM)
#define PAGE_SCALE_BYTES  (PAGE_SIZE * (int)sizeof(float))
#define PAGE_STRIDE       (PAGE_DATA_BYTES + PAGE_SCALE_BYTES)

#define SMEM_STRIDE  (HEAD_DIM + 1)






//Q is stored as [batch, num_heads * head_dim]

__global__ void compute_scores_kernel(
    const float*   __restrict__ Q,
    const uint8_t* __restrict__ K_cache,
    const float*   __restrict__ head_weights,
    float*         __restrict__ scores,
    const int*     __restrict__ page_table,
    int max_seq_len,
    int max_pages
) {
/**************************** Initialize row pointers & Shared memory***************************************************** */
    //      (CURRENT BLOCK COLUMN)
    int batch_idx = blockIdx.x;

    //      (CURRENT BLOCK ROW) * (SIZE OF ROW) + (POSITION WITHIN ROW)
    int token_idx = blockIdx.y * blockDim.x + threadIdx.x;

    // Stride to go from section of heads to next
    //              (NUMBER OF HEADS) * (SIZE OF HEAD IN SMEM)
    __shared__ float smem_q[NUM_HEADS * SMEM_STRIDE];

    // Current location within matrix Q. This holds two dimensions NUM_HEADS and HEAD_DIM. We require / to access NUM_HEADS and % to access HEAD_DIM.
    //
    //              (BASE POINTER) + (BATCH COLUMN) ()
    const float* q_base = Q + batch_idx * NUM_HEADS * HEAD_DIM;


/**************************** LOAD Q into SMEM_Q***************************************************** */

    ///What does each thread contribute to in our paged diemnsion grid strided!!
    // This is essentially pre-mapping our thread to our paged dimension
    // Its like a piano where the keys are the threads, and they map to outer dimensions.
    // It doens't cost to traverse this outer dimension because the keys themselves are tied to them
    for (int i = threadIdx.x; i < NUM_HEADS * HEAD_DIM; i += blockDim.x) {
        // Which batch are we in (row)
        int h = i / HEAD_DIM;
        // Which token are we within the batch (column)
        int d = i % HEAD_DIM;

        // Store current value in q_base in SMEM where h traverses row at size SMEM_STRIDE and d is our column within our row
        // (CURRENT ROW) * (SIZE OF ROW IN SMEM) + (COLUMN) = CURRENT VALUE IN Q
        // q_base pointer becomes q_base = Q + batch_idx * NUM_HEADS * HEAD_DIM + i;
        // We traverse over NUM_HEADS * HEAD_DIM because we already grabbed those values using / (NUM_HEADS) and % (HEAD_DIM)
        smem_q[h * SMEM_STRIDE + d] = q_base[i];
    }

    // We are moving frame data moving, to data processing
    __syncthreads();

    // If no more to process, exit
    if (token_idx >= max_seq_len) return;

/**************************** LOAD and scale our K_cache ***************************************************** */

    // Our current row
    int logical_page = token_idx / PAGE_SIZE;
    // Our current column
    int offset = token_idx % PAGE_SIZE;

    //If our 2D page goes beyond the number of pages
    if (logical_page >= max_pages) {
        // Set current token to 0.0f in our output matrix
        //   (CURRENT BATCH) * (SIZE OF BATCH) + (CURRENT TOKEN)
        scores[batch_idx * max_seq_len + token_idx] = 0.0f;
        return;
    }

    // Paging, similar to KV Cache
    // Our current physical page index
    //                          (CURRENT BATCH) * (NUMBER OF PAGES) + (CURRENT PAGE)
    int physical_page = page_table[batch_idx * max_pages + logical_page];

    // Our location in K_Cache (page)
    //    (BASE LOCATION WITHIN K_CACHE) + (SIZE OF PAGE IN BYTES) * (CONTENTS OF PAGE SIZE)
    const uint8_t* page_base = K_cache + (size_t)physical_page * PAGE_STRIDE;

    // Our location within our page (token)
    //                   (PAGE BASE) + (SIZE OF TOKEN IN BYTES) * (CONTENTS OF TOKEN SIZE)
    const uint8_t* k_fp8 = page_base + (size_t)offset * HEAD_DIM;

    // The bytes we are about to read our FP32, hence we indicate that to the compiler
    float k_scale = *reinterpret_cast<const float*>(
        // Skip past all FP8 values, reach our scales, offset within scales
        // (PAGE_BASE) + (SIZE OF PAGE * SIZE OF HEAD) + (CURRENT TOKEN) * (SIZE_OF FLOAT IN BYTES)
        page_base + PAGE_DATA_BYTES + offset * sizeof(float)
    );

    // How many head each token has
    float dots[NUM_HEADS];

    // Ensure our dots array is initialized to 0.0f
    // Can be done at compile time
    #pragma unroll
    for (int h = 0; h < NUM_HEADS; h++) {
        dots[h] = 0.0f;
    }


    // Iterate through each one of our elements
    for (int d = 0; d < HEAD_DIM; d++) {
        // Our FP8 value
        __nv_fp8_e4m3 fp8_val;

        // Copy our FP8 from our K cache, and copy it into our FP8 value
        //        To       From                Size 
        memcpy(&fp8_val, &k_fp8[d], sizeof(__nv_fp8_e4m3));
        
        //Convert our fp8_value into FP32 * our scale
        float kd = __half2float((__half)fp8_val) * k_scale;

/**************************** COMPUTE our Q and K dot product ***************************************************** */
        //Allow our loop to be at compile time
        #pragma unroll
        //Iterate over the number of heads
        for (int h = 0; h < NUM_HEADS; h++) {
            //DOT Product of our Q values and our K values
            dots[h] += smem_q[h * SMEM_STRIDE + d] * kd;
        }
    }


/**************************** COMPUTE, Apply our activation function to our Q and K simliarity scores ***************************************************** */
    
    float total_score = 0.0f;
    #pragma unroll
    
    //Iterate over our heads
    for (int h = 0; h < NUM_HEADS; h++) {
        // Our activation function
        // Prevents our values from collapsing into a single A * B + C dimension, allows for more expression

        float relu_dot;
        if (dots[h] > 0.0f) {
            relu_dot = dots[h];  // Keep the value if it's positive
        } else {
            relu_dot = 0.0f;     // If not, set to zero
        }

        //DOT Product our ReLU'd input with our head weights (input * weight)
        total_score += relu_dot * head_weights[h];
    }

/**************************** STORE our activated Q and K similarity scores ***************************************************** */

    // Store our total score in our scores tensor [batch_size,max_seq_len]
    scores[batch_idx * max_seq_len + token_idx] = total_score;
}













// CODE BELOW IS AI GENERATED 










// ─────────────────────────────────────────────────────────────────────────────
//  GPU TopK — for each batch row, returns k global token indices sorted
//  descending by score.  Uses Thrust sort-by-key so it is correct but not
//  competition-speed; replace with CUB DeviceSegmentedRadixSort for Week 8.
//
//  d_scores:      device float [batch_size, max_seq_len]
//  d_topk_out:    device int   [batch_size, k]  — OUTPUT (global indices)
// ─────────────────────────────────────────────────────────────────────────────
static void gpu_topk(
    const float* d_scores,
    int*         d_topk_out,
    int batch_size, int max_seq_len, int k
) {
    for (int b = 0; b < batch_size; b++) {
        // Grab a view of this batch's score row
        const float* row_ptr = d_scores + b * max_seq_len;

        // Copy scores into a sortable device vector (we need a mutable copy)
        thrust::device_vector<float> d_vals(row_ptr, row_ptr + max_seq_len);

        // Fill indices 0 … max_seq_len-1 — these are the GLOBAL token indices
        thrust::device_vector<int> d_idx(max_seq_len);
        thrust::sequence(d_idx.begin(), d_idx.end());   // 0, 1, 2, …

        // Sort (score, index) pairs descending by score.
        // After this, d_idx[0] is the index of the highest-scoring token.
        // This is where sorting direction is enforced: thrust::greater<float>()
        // means descending.  Using thrust::less<float>() here would silently
        // return the LOWEST scores — the sorting-direction bug.
        thrust::sort_by_key(
            d_vals.begin(), d_vals.end(),
            d_idx.begin(),
            thrust::greater<float>()
        );

        // Copy the first k indices to the output — these are global indices
        // (values 0 … max_seq_len-1), NOT page-local offsets (0 … PAGE_SIZE-1).
        thrust::copy_n(
            d_idx.begin(), k,
            thrust::device_pointer_cast(d_topk_out + b * k)
        );
    }
}


// ─────────────────────────────────────────────────────────────────────────────
//  CPU reference implementation — used for correctness comparison
// ─────────────────────────────────────────────────────────────────────────────
static void compute_scores_cpu(
    const float*   Q,
    const uint8_t* K_cache,
    const float*   head_weights,
    float*         scores,
    const int*     page_table,
    int batch_size, int max_seq_len, int max_pages
) {
    for (int b = 0; b < batch_size; b++) {
        for (int tok = 0; tok < max_seq_len; tok++) {
            int logical_page = tok / PAGE_SIZE;
            int offset       = tok % PAGE_SIZE;

            if (logical_page >= max_pages) {
                scores[b * max_seq_len + tok] = 0.0f;
                continue;
            }

            int phys = page_table[b * max_pages + logical_page];
            const uint8_t* page_base = K_cache + (size_t)phys * PAGE_STRIDE;
            const uint8_t* k_fp8     = page_base + (size_t)offset * HEAD_DIM;
            float k_scale = *reinterpret_cast<const float*>(
                page_base + PAGE_DATA_BYTES + offset * sizeof(float));

            const float* q_base = Q + b * NUM_HEADS * HEAD_DIM;

            float total = 0.0f;
            for (int h = 0; h < NUM_HEADS; h++) {
                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) {
                    __nv_fp8_e4m3 fp8_val;
                    memcpy(&fp8_val, &k_fp8[d], sizeof(__nv_fp8_e4m3));
                    float kd = __half2float((__half)fp8_val) * k_scale;
                    dot += q_base[h * HEAD_DIM + d] * kd;
                }
                total += ((dot > 0.0f) ? dot : 0.0f) * head_weights[h];
            }
            scores[b * max_seq_len + tok] = total;
        }
    }
}


// ─────────────────────────────────────────────────────────────────────────────
//  Test utilities
// ─────────────────────────────────────────────────────────────────────────────
static void fill_rand(float* buf, int n) {
    for (int i = 0; i < n; i++) buf[i] = ((float)rand() / RAND_MAX) - 0.5f;
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

// Build a K cache with uniform float value per page.
// page_vals[pg] sets the FP8 value stored for every byte in that page's data
// section.  Scales are all 1.0f so k_scale has no effect on the dot product.
static uint8_t* build_k_cache_custom(int num_phys_pages, const float* page_vals) {
    size_t total = (size_t)num_phys_pages * PAGE_STRIDE;
    uint8_t* buf = (uint8_t*)malloc(total);
    for (int pg = 0; pg < num_phys_pages; pg++) {
        uint8_t* page  = buf + (size_t)pg * PAGE_STRIDE;
        uint8_t  byte  = float_to_fp8_byte(page_vals[pg]);
        memset(page, byte, PAGE_DATA_BYTES);
        float* scales = reinterpret_cast<float*>(page + PAGE_DATA_BYTES);
        for (int t = 0; t < PAGE_SIZE; t++) scales[t] = 1.0f;
    }
    return buf;
}

// Standard random K cache used by original tests
static uint8_t* build_synthetic_k_cache(int num_phys_pages) {
    size_t total = (size_t)num_phys_pages * PAGE_STRIDE;
    uint8_t* buf = (uint8_t*)malloc(total);
    for (int pg = 0; pg < num_phys_pages; pg++) {
        uint8_t* page = buf + (size_t)pg * PAGE_STRIDE;
        for (int i = 0; i < PAGE_SIZE * HEAD_DIM; i++) {
            float v = ((float)rand() / RAND_MAX) - 0.5f;
            page[i] = float_to_fp8_byte(v);
        }
        float* scales = reinterpret_cast<float*>(page + PAGE_DATA_BYTES);
        for (int t = 0; t < PAGE_SIZE; t++) scales[t] = 1.0f;
    }
    return buf;
}

// Convenience: launch scoring kernel and synchronize
static void launch_scores(
    float* d_Q, uint8_t* d_K, float* d_weights, float* d_scores, int* d_pt,
    int batch_size, int max_seq_len, int max_pages
) {
    constexpr int THREADS = 256;
    dim3 grid(batch_size, (max_seq_len + THREADS - 1) / THREADS);
    compute_scores_kernel<<<grid, THREADS>>>(
        d_Q, d_K, d_weights, d_scores, d_pt, max_seq_len, max_pages);
    cudaDeviceSynchronize();
}


// ─────────────────────────────────────────────────────────────────────────────
//  Original Test 1 — score correctness (identity page table, random data)
// ─────────────────────────────────────────────────────────────────────────────
static void test_score_correctness() {
    printf("\n=== test_score_correctness ===\n");

    const int batch_size  = 2;
    const int num_pages   = 4;
    const int max_seq_len = num_pages * PAGE_SIZE;

    int q_elems = batch_size * NUM_HEADS * HEAD_DIM;
    float* h_Q = (float*)malloc(q_elems * sizeof(float));
    fill_rand(h_Q, q_elems);

    float* h_weights = (float*)malloc(NUM_HEADS * sizeof(float));
    fill_rand(h_weights, NUM_HEADS);
    for (int i = 0; i < NUM_HEADS; i++) h_weights[i] = fabsf(h_weights[i]);

    uint8_t* h_K = build_synthetic_k_cache(num_pages);

    int* h_pt = (int*)malloc(batch_size * num_pages * sizeof(int));
    for (int b = 0; b < batch_size; b++)
        for (int p = 0; p < num_pages; p++)
            h_pt[b * num_pages + p] = p;

    float* h_scores_gpu = (float*)malloc(batch_size * max_seq_len * sizeof(float));
    float* h_scores_cpu = (float*)malloc(batch_size * max_seq_len * sizeof(float));
    compute_scores_cpu(h_Q, h_K, h_weights, h_scores_cpu, h_pt,
                       batch_size, max_seq_len, num_pages);

    float *d_Q, *d_weights, *d_scores;
    uint8_t *d_K;
    int *d_pt;
    cudaMalloc(&d_Q,       q_elems * sizeof(float));
    cudaMalloc(&d_K,       (size_t)num_pages * PAGE_STRIDE);
    cudaMalloc(&d_weights, NUM_HEADS * sizeof(float));
    cudaMalloc(&d_scores,  batch_size * max_seq_len * sizeof(float));
    cudaMalloc(&d_pt,      batch_size * num_pages * sizeof(int));
    cudaMemcpy(d_Q,       h_Q,       q_elems * sizeof(float),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_K,       h_K,       (size_t)num_pages * PAGE_STRIDE,      cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, NUM_HEADS * sizeof(float),            cudaMemcpyHostToDevice);
    cudaMemcpy(d_pt,      h_pt,      batch_size * num_pages * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_scores, 0, batch_size * max_seq_len * sizeof(float));

    launch_scores(d_Q, d_K, d_weights, d_scores, d_pt, batch_size, max_seq_len, num_pages);
    cudaMemcpy(h_scores_gpu, d_scores, batch_size * max_seq_len * sizeof(float), cudaMemcpyDeviceToHost);

    float err = max_abs_error(h_scores_gpu, h_scores_cpu, batch_size * max_seq_len);
    printf("Score correctness: max_err = %.2e  %s\n", err, err < 1e-3f ? "PASS" : "FAIL");

    free(h_Q); free(h_weights); free(h_K); free(h_pt);
    free(h_scores_gpu); free(h_scores_cpu);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_weights); cudaFree(d_scores); cudaFree(d_pt);
}


// ─────────────────────────────────────────────────────────────────────────────
//  Original Test 2 — paged addressing (reversed page table)
// ─────────────────────────────────────────────────────────────────────────────
static void test_paged_addressing() {
    printf("\n=== test_paged_addressing (shuffled page table) ===\n");

    const int batch_size  = 1;
    const int num_pages   = 8;
    const int max_seq_len = num_pages * PAGE_SIZE;

    int q_elems = batch_size * NUM_HEADS * HEAD_DIM;
    float* h_Q = (float*)malloc(q_elems * sizeof(float));
    fill_rand(h_Q, q_elems);

    float* h_weights = (float*)malloc(NUM_HEADS * sizeof(float));
    for (int i = 0; i < NUM_HEADS; i++) h_weights[i] = fabsf((float)rand() / RAND_MAX);

    uint8_t* h_K = build_synthetic_k_cache(num_pages);

    int* h_pt = (int*)malloc(num_pages * sizeof(int));
    for (int p = 0; p < num_pages; p++) h_pt[p] = num_pages - 1 - p;

    float* h_scores_gpu = (float*)malloc(max_seq_len * sizeof(float));
    float* h_scores_cpu = (float*)malloc(max_seq_len * sizeof(float));
    compute_scores_cpu(h_Q, h_K, h_weights, h_scores_cpu, h_pt, 1, max_seq_len, num_pages);

    float *d_Q, *d_weights, *d_scores;
    uint8_t *d_K;
    int *d_pt;
    cudaMalloc(&d_Q,       q_elems * sizeof(float));
    cudaMalloc(&d_K,       (size_t)num_pages * PAGE_STRIDE);
    cudaMalloc(&d_weights, NUM_HEADS * sizeof(float));
    cudaMalloc(&d_scores,  max_seq_len * sizeof(float));
    cudaMalloc(&d_pt,      num_pages * sizeof(int));
    cudaMemcpy(d_Q,       h_Q,       q_elems * sizeof(float),         cudaMemcpyHostToDevice);
    cudaMemcpy(d_K,       h_K,       (size_t)num_pages * PAGE_STRIDE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, NUM_HEADS * sizeof(float),       cudaMemcpyHostToDevice);
    cudaMemcpy(d_pt,      h_pt,      num_pages * sizeof(int),         cudaMemcpyHostToDevice);
    cudaMemset(d_scores, 0, max_seq_len * sizeof(float));

    launch_scores(d_Q, d_K, d_weights, d_scores, d_pt, 1, max_seq_len, num_pages);
    cudaMemcpy(h_scores_gpu, d_scores, max_seq_len * sizeof(float), cudaMemcpyDeviceToHost);

    float err = max_abs_error(h_scores_gpu, h_scores_cpu, max_seq_len);
    printf("Paged addressing:  max_err = %.2e  %s\n", err, err < 1e-3f ? "PASS" : "FAIL");

    free(h_Q); free(h_weights); free(h_K); free(h_pt);
    free(h_scores_gpu); free(h_scores_cpu);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_weights); cudaFree(d_scores); cudaFree(d_pt);
}


// ─────────────────────────────────────────────────────────────────────────────
//  Original Test 3 — TopK ranking logic (CPU only)
// ─────────────────────────────────────────────────────────────────────────────
static void test_topk_ranking() {
    printf("\n=== test_topk_ranking (CPU ranking logic) ===\n");

    const int seq_len = 512;
    const int k       = 16;

    float* scores = (float*)malloc(seq_len * sizeof(float));
    fill_rand(scores, seq_len);

    int* topk = (int*)malloc(k * sizeof(int));
    for (int i = 0; i < k; i++) topk[i] = -1;
    float* topk_vals = (float*)malloc(k * sizeof(float));
    for (int i = 0; i < k; i++) topk_vals[i] = -INFINITY;

    for (int i = 0; i < seq_len; i++) {
        if (scores[i] > topk_vals[k - 1]) {
            topk_vals[k - 1] = scores[i];
            topk[k - 1]      = i;
            for (int j = k - 1; j > 0 && topk_vals[j] > topk_vals[j - 1]; j--) {
                float tv = topk_vals[j]; topk_vals[j] = topk_vals[j-1]; topk_vals[j-1] = tv;
                int   ti = topk[j];      topk[j] = topk[j-1]; topk[j-1] = ti;
            }
        }
    }

    float min_selected = topk_vals[k - 1];
    int errors = 0;
    for (int i = 0; i < seq_len; i++) {
        int in_topk = 0;
        for (int j = 0; j < k; j++) if (topk[j] == i) { in_topk = 1; break; }
        if (!in_topk && scores[i] > min_selected) errors++;
    }

    printf("Top-k ranking:     errors = %d  %s\n", errors, errors == 0 ? "PASS" : "FAIL");
    free(scores); free(topk); free(topk_vals);
}


// ─────────────────────────────────────────────────────────────────────────────
//  Original Test 4 — batch independence
// ─────────────────────────────────────────────────────────────────────────────
static void test_batch_independence() {
    printf("\n=== test_batch_independence ===\n");

    const int batch_size  = 2;
    const int num_pages   = 4;
    const int max_seq_len = num_pages * PAGE_SIZE;

    int q_elems = batch_size * NUM_HEADS * HEAD_DIM;
    float* h_Q = (float*)malloc(q_elems * sizeof(float));
    for (int i = 0; i < NUM_HEADS * HEAD_DIM; i++) h_Q[i]                       =  0.5f;
    for (int i = 0; i < NUM_HEADS * HEAD_DIM; i++) h_Q[NUM_HEADS * HEAD_DIM + i] = -0.5f;

    float* h_weights = (float*)malloc(NUM_HEADS * sizeof(float));
    for (int i = 0; i < NUM_HEADS; i++) h_weights[i] = 1.0f / NUM_HEADS;

    uint8_t* h_K = build_synthetic_k_cache(num_pages);

    int* h_pt = (int*)malloc(batch_size * num_pages * sizeof(int));
    for (int b = 0; b < batch_size; b++)
        for (int p = 0; p < num_pages; p++)
            h_pt[b * num_pages + p] = p;

    float* h_scores = (float*)malloc(batch_size * max_seq_len * sizeof(float));

    float *d_Q, *d_weights, *d_scores;
    uint8_t *d_K;
    int *d_pt;
    cudaMalloc(&d_Q,       q_elems * sizeof(float));
    cudaMalloc(&d_K,       (size_t)num_pages * PAGE_STRIDE);
    cudaMalloc(&d_weights, NUM_HEADS * sizeof(float));
    cudaMalloc(&d_scores,  batch_size * max_seq_len * sizeof(float));
    cudaMalloc(&d_pt,      batch_size * num_pages * sizeof(int));
    cudaMemcpy(d_Q,       h_Q,       q_elems * sizeof(float),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_K,       h_K,       (size_t)num_pages * PAGE_STRIDE,      cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, NUM_HEADS * sizeof(float),            cudaMemcpyHostToDevice);
    cudaMemcpy(d_pt,      h_pt,      batch_size * num_pages * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_scores, 0, batch_size * max_seq_len * sizeof(float));

    launch_scores(d_Q, d_K, d_weights, d_scores, d_pt, batch_size, max_seq_len, num_pages);
    cudaMemcpy(h_scores, d_scores, batch_size * max_seq_len * sizeof(float), cudaMemcpyDeviceToHost);

    float sum0 = 0.f, sum1 = 0.f;
    for (int i = 0; i < max_seq_len; i++) sum0 += h_scores[i];
    for (int i = 0; i < max_seq_len; i++) sum1 += h_scores[max_seq_len + i];

    printf("Batch independence: sum[0]=%.2f  sum[1]=%.2f  %s\n",
           sum0, sum1, (sum0 != sum1) ? "PASS" : "FAIL");

    free(h_Q); free(h_weights); free(h_K); free(h_pt); free(h_scores);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_weights); cudaFree(d_scores); cudaFree(d_pt);
}


// ─────────────────────────────────────────────────────────────────────────────
//  NEW Test 5 — index space correctness
//
//  The silent killer: does gpu_topk return GLOBAL token indices (0…seq_len-1),
//  or PAGE-LOCAL offsets (0…PAGE_SIZE-1)?
//
//  Design: 2 pages.  Page 0 K values = 0.0 (all-zero FP8).  Page 1 K values =
//  1.0 (round-trips FP8 exactly).  Q = all +1.0.  With scale=1.0 and positive
//  Q, only page-1 tokens produce non-zero scores.
//
//  If indices are global:  all returned indices are in [64, 127].
//  If indices are page-local: returned indices would incorrectly be in [0, 63],
//  because offset=0…63 within page 1 would be mistaken for global token index.
// ─────────────────────────────────────────────────────────────────────────────
static void test_index_space_correctness() {
    printf("\n=== test_index_space_correctness ===\n");

    const int batch_size  = 1;
    const int num_pages   = 2;   // page 0: tokens 0-63, page 1: tokens 64-127
    const int max_seq_len = num_pages * PAGE_SIZE;  // 128
    const int k           = PAGE_SIZE;              // ask for 64 results — exactly page 1

    // Q = all +1.0 so every dot product is simply sum of K values
    float* h_Q = (float*)malloc(NUM_HEADS * HEAD_DIM * sizeof(float));
    for (int i = 0; i < NUM_HEADS * HEAD_DIM; i++) h_Q[i] = 1.0f;

    // All head weights = 1.0 / NUM_HEADS so total_score = mean dot product across heads
    float* h_weights = (float*)malloc(NUM_HEADS * sizeof(float));
    for (int i = 0; i < NUM_HEADS; i++) h_weights[i] = 1.0f / NUM_HEADS;

    // page_vals[0]=0.0 → page 0 K=0, dots=0, score=0
    // page_vals[1]=1.0 → page 1 K=1 (FP8-exact), dots=128, score=128
    float page_vals[2] = { 0.0f, 1.0f };
    uint8_t* h_K = build_k_cache_custom(num_pages, page_vals);

    // Identity page table: logical page p → physical page p
    int h_pt[2] = { 0, 1 };

    float *d_Q, *d_weights, *d_scores;
    uint8_t *d_K;
    int *d_pt, *d_topk;
    cudaMalloc(&d_Q,       NUM_HEADS * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_K,       (size_t)num_pages * PAGE_STRIDE);
    cudaMalloc(&d_weights, NUM_HEADS * sizeof(float));
    cudaMalloc(&d_scores,  max_seq_len * sizeof(float));
    cudaMalloc(&d_pt,      num_pages * sizeof(int));
    cudaMalloc(&d_topk,    k * sizeof(int));

    cudaMemcpy(d_Q,       h_Q,       NUM_HEADS * HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K,       h_K,       (size_t)num_pages * PAGE_STRIDE,      cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, NUM_HEADS * sizeof(float),            cudaMemcpyHostToDevice);
    cudaMemcpy(d_pt,      h_pt,      num_pages * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemset(d_scores, 0, max_seq_len * sizeof(float));

    launch_scores(d_Q, d_K, d_weights, d_scores, d_pt, 1, max_seq_len, num_pages);
    gpu_topk(d_scores, d_topk, 1, max_seq_len, k);

    int* h_topk = (int*)malloc(k * sizeof(int));
    cudaMemcpy(h_topk, d_topk, k * sizeof(int), cudaMemcpyDeviceToHost);

    // Every returned index must be >= PAGE_SIZE (i.e., a global index into page 1).
    // If even one index is < PAGE_SIZE it means the kernel returned a page-local
    // offset instead of a global token index — the index-space bug.
    int errors = 0;
    for (int i = 0; i < k; i++) {
        if (h_topk[i] < PAGE_SIZE) {
            errors++;
            if (errors <= 3)  // print first few to aid debugging
                printf("  index-space BUG: topk[%d]=%d (should be >= %d)\n",
                       i, h_topk[i], PAGE_SIZE);
        }
    }
    printf("Index space:        errors = %d  %s\n", errors, errors == 0 ? "PASS" : "FAIL");

    free(h_Q); free(h_weights); free(h_K); free(h_topk);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_weights);
    cudaFree(d_scores); cudaFree(d_pt); cudaFree(d_topk);
}


// ─────────────────────────────────────────────────────────────────────────────
//  NEW Test 6 — sorting direction
//
//  The silent killer: does gpu_topk return the HIGHEST-scoring tokens, or the
//  lowest?  An ascending sort would give you the wrong end of the ranking with
//  no crash and no obvious error message.
//
//  Design: inject scores[i] = (float)i directly — no scoring kernel involved.
//  Top-k descending must return the k largest indices: [seq_len-1, ..., seq_len-k].
//  Ascending sort would return [0, 1, ..., k-1] instead.
//  We verify by checking that every returned index belongs to [seq_len-k, seq_len-1].
// ─────────────────────────────────────────────────────────────────────────────
static void test_sorting_direction() {
    printf("\n=== test_sorting_direction ===\n");

    const int seq_len = 512;
    const int k       = 32;

    // scores[i] = i, so the k highest are indices [seq_len-k … seq_len-1]
    float* h_scores = (float*)malloc(seq_len * sizeof(float));
    for (int i = 0; i < seq_len; i++) h_scores[i] = (float)i;

    float* d_scores;
    int*   d_topk;
    cudaMalloc(&d_scores, seq_len * sizeof(float));
    cudaMalloc(&d_topk,   k * sizeof(int));
    cudaMemcpy(d_scores, h_scores, seq_len * sizeof(float), cudaMemcpyHostToDevice);

    // Run GPU TopK — this is the operation under test, not the scoring kernel
    gpu_topk(d_scores, d_topk, /*batch_size=*/1, seq_len, k);

    int* h_topk = (int*)malloc(k * sizeof(int));
    cudaMemcpy(h_topk, d_topk, k * sizeof(int), cudaMemcpyDeviceToHost);

    // Every returned index must be in the top-k score band [seq_len-k, seq_len-1].
    // If gpu_topk sorted ascending, we'd get indices 0…k-1 here instead.
    int errors = 0;
    for (int i = 0; i < k; i++) {
        if (h_topk[i] < seq_len - k) {
            errors++;
            if (errors <= 3)
                printf("  direction BUG: topk[%d]=%d  score=%.0f  (expected index >= %d)\n",
                       i, h_topk[i], h_scores[h_topk[i]], seq_len - k);
        }
    }
    printf("Sorting direction:  errors = %d  %s\n", errors, errors == 0 ? "PASS" : "FAIL");

    free(h_scores); free(h_topk);
    cudaFree(d_scores); cudaFree(d_topk);
}


// ─────────────────────────────────────────────────────────────────────────────
//  NEW Test 7 — tiny manually-verifiable example
//
//  The principle from Week 6: "test with tiny examples where you can manually
//  verify the expected output."
//
//  Design: batch=1, seq_len=256 (4 pages), k=4.  We bypass the scoring kernel
//  and inject known scores directly: exactly 4 tokens get score 100.0, all
//  others get score 0.0.  We know by construction which 4 indices are correct.
//  You can verify this test by reading the expected_indices array below.
// ─────────────────────────────────────────────────────────────────────────────
static void test_tiny_manual() {
    printf("\n=== test_tiny_manual (hand-verifiable) ===\n");

    const int seq_len = 256;
    const int k       = 4;

    // These 4 indices are the expected top-k — chosen to span multiple pages
    // so we also exercise the page boundary logic implicitly.
    // Token  42 lives in page 0 (offset  42)
    // Token  77 lives in page 1 (offset  13)
    // Token 133 lives in page 2 (offset   5)
    // Token 201 lives in page 3 (offset   9)
    const int expected_indices[4] = { 42, 77, 133, 201 };

    float* h_scores = (float*)calloc(seq_len, sizeof(float)); // all 0.0
    for (int i = 0; i < k; i++)
        h_scores[expected_indices[i]] = 100.0f;

    float* d_scores;
    int*   d_topk;
    cudaMalloc(&d_scores, seq_len * sizeof(float));
    cudaMalloc(&d_topk,   k * sizeof(int));
    cudaMemcpy(d_scores, h_scores, seq_len * sizeof(float), cudaMemcpyHostToDevice);

    gpu_topk(d_scores, d_topk, /*batch_size=*/1, seq_len, k);

    int* h_topk = (int*)malloc(k * sizeof(int));
    cudaMemcpy(h_topk, d_topk, k * sizeof(int), cudaMemcpyDeviceToHost);

    // Build a sorted set of returned indices so comparison is order-independent
    int returned_sorted[4];
    memcpy(returned_sorted, h_topk, k * sizeof(int));
    std::sort(returned_sorted, returned_sorted + k);

    int expected_sorted[4];
    memcpy(expected_sorted, expected_indices, k * sizeof(int));
    std::sort(expected_sorted, expected_sorted + k);

    int errors = 0;
    for (int i = 0; i < k; i++) {
        if (returned_sorted[i] != expected_sorted[i]) {
            errors++;
            printf("  mismatch at rank %d: got %d  expected %d\n",
                   i, returned_sorted[i], expected_sorted[i]);
        }
    }
    printf("Tiny manual:        errors = %d  %s\n", errors, errors == 0 ? "PASS" : "FAIL");
    printf("  expected indices: {%d, %d, %d, %d}\n",
           expected_indices[0], expected_indices[1],
           expected_indices[2], expected_indices[3]);
    printf("  returned indices: {%d, %d, %d, %d}\n",
           returned_sorted[0], returned_sorted[1],
           returned_sorted[2], returned_sorted[3]);

    free(h_scores); free(h_topk);
    cudaFree(d_scores); cudaFree(d_topk);
}


// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    srand(42);
    printf("========== Week 6: Top-K Indexer (local RTX 4060 tests) ==========\n");

    // Original tests — validate scoring kernel
    test_score_correctness();
    test_paged_addressing();
    test_topk_ranking();
    test_batch_independence();

    // New tests — validate the three silent killers from Week 6 spec
    test_index_space_correctness();   // global vs page-local indices
    test_sorting_direction();         // descending vs ascending sort
    test_tiny_manual();               // hand-verifiable tiny case

    printf("\n===================================================================\n");
    return 0;
}
