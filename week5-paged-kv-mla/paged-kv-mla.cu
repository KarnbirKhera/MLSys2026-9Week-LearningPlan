#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>

#define TILE_SIZE 16
#define PAGE_SIZE 64


/*
PAGED ATTENTION

Rather than using Coordinate * Stride + Offset to go from our logical indexes -> physical address, we use a paged system.
This allows for efficient memory allocation.

List of assumptions:
- Q is not a square matrix
- K is not a square matrix
- V is not a square matrix
- O is not a square matrix

    DIMENSIONS OF OUR INPUT
    Q -> (seq_q,d_head)
    K -> (seq_k, d_head)
    V -> (seq_k, d_head)
    O -> (seq_q, d_head)



*/

__global__ void attention_paged_v1(
    const float* __restrict__ Q,
    const float* __restrict__ K_cache,
    const float* __restrict__ V_cache,
    const int*   __restrict__ page_table,
    float*       __restrict__ O,
    float*       __restrict__ out_lse,
    int batch_size, int num_heads,
    int seq_q, int seq_k, int d_head,
    int max_logical_pages,
    float constant_scale
) {
    /**************************** Initialize row pointers & Shared memory***************************************************** */
    __shared__ float Q_tile[TILE_SIZE][TILE_SIZE];
    // We access K_tile column wise for our implicit transpose for Q * K
    __shared__ float K_tile[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float V_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float S_tile[TILE_SIZE][TILE_SIZE];

    // Our current row/column in the Z dimension, we are going from a 1D address to a 2D index
    // Each row is a batch, within that row, is our heads
    int batch_idx = blockIdx.z / num_heads;
    int head_idx = blockIdx.z % num_heads;


    // Stride needed to go from head to head within our current batch (column to column)
    int head_stride_q_o = seq_q * d_head;
    //Stride needed to go batch from batch (row to row)
    int batch_stride_q_o = num_heads * head_stride_q_o;

    // Our global position within our Q matrice
    //                 (Which Batch) * (Size of Batch) + (Which head) * (Size of Head) + Q offset
    const float* Q_head = batch_idx * batch_stride_q_o + head_idx * head_stride_q_o + Q;
    
    // Our global position within O matrice
    //                 (Which Batch) * (Size of Batch) + (Which head) * (Size of Head) + O offset
    float* O_head = batch_idx * batch_stride_q_o + head_idx * head_stride_q_o + O;



    //                        (Which batch) * (Size of Batch) + (Where i Batch)
    const int* this_page_table = batch_idx * max_logical_pages + page_table;

    // Which row are we in
    //        (Which Block) * (Size of Block) + (Which thread within block row wise)
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    // Which column are we in
    //       (Which Block) * (Size of Block) + (Which thread within block column wise)
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    //Initialize our softmax numerator, max num and runningTotal. We use infinity as to ensure our number is always greater than our intialized value
    float softmax_numerator = 0.0f;
    float max_number = -INFINITY;
    float runningSum = 0.0f;

    // How many times does our tile need to slide across our K matrice (long side)
    int num_k_tiles = (seq_k + TILE_SIZE - 1) / TILE_SIZE;

    for (int tileId_LongSide = 0; tileId_LongSide < num_k_tiles; tileId_LongSide++) {

        //Ensure we start with 0.0f to later be accumulated, also resetting our last tile's value
        S_tile[threadIdx.y][threadIdx.x] = 0.0f;

        // How many times does our tile need to slide across our common d_head dimension (short side)
        int num_d_tiles = (d_head + TILE_SIZE - 1) / TILE_SIZE;

        for (int tileId_ShortSide = 0; tileId_ShortSide < num_d_tiles; tileId_ShortSide++) {

    /**************************** Load our Q matrice into Q_tile shared memory***************************************************** */
            // Our current column within the Q matrice, iterator
            int q_col = tileId_ShortSide * TILE_SIZE + threadIdx.x;

            //Bounds check
            if (row < seq_q && q_col < d_head)
                //Input out Q_head data into Q_tile (no transpose)
                //                          (Which row) * (How big is the row) + (Which column within the row)
                Q_tile[threadIdx.y][threadIdx.x] = Q_head[row * d_head + q_col];
            else
                //Pad 0.0f if out of bounds. When we calculate later, we will have no need to worry about our out of bounds effecting our calculations
                Q_tile[threadIdx.y][threadIdx.x] = 0.0f;

    /**************************** Load our K matrice into K_tile shared memory***************************************************** */
            // Our current column within our K matrice, iterator
            int k_d_idx   = tileId_ShortSide * TILE_SIZE + threadIdx.x;
            //Our current row within our K matrice, iterator and parallelized
            int k_seq_idx = tileId_LongSide  * TILE_SIZE + threadIdx.y;




            /**
             * ------------------------------------------------------------------
             *       Matrix A                              Matrix B
             *        M x K                                 K x N
             * ------------------------------------------------------------------
             *  Which row * rowsize + col             Which row * rowsize + col
             * 
             *       seq_k                                     
             *  |.------------|                          .--------------.
             *  |=[k_seq_idx]=>                          ||v|---------- | 
            *   |-------------|                  d_head  ||v|-----------|
             *  |-------------|                          ||k_d_idx|-----|
             *  |-------------|                          ||v|-----------|
             *  '-------------'                          '--------------'
             *                                            
             *                                            
             *
             *  To ensure our column (k_seq_idx) indexer doesnt go out of bounds, we use our seq_k size variable
             *  To ensure our row (k_d_idx) indexer doesnt go out of bounds, we use our d_head size variable 
             * ------------------------------------------------------------------
             */
             //Out of bounds check. A diagram to help us see this why we use these variables 
            if (k_seq_idx < seq_k && k_d_idx < d_head) {
                // We are doing a 1D hardware address to a 2D logical index conversion
                // Our / gives us the row
                // Our % gives us the column





                //
                // For our dimensional analysis, we have the following order
                // page -> tokens -> heads -> element
                //

                // Gives us the row, which page are we at (logical 2D index)
                int logical_page  = k_seq_idx / PAGE_SIZE;
                // Gives us the column, which token are we at in our page
                int token_in_page = k_seq_idx % PAGE_SIZE;

                // What page are we at (row) (physical 1D page index)
                int phys_page = this_page_table[logical_page];
                
                // How many Elements per Page
                //               (Tokens per Page) * (Heads per Token) * (Element per Head)
                int page_stride  = PAGE_SIZE * num_heads * d_head;

                // How many elements per Token
                //               (Heads per Token) * (Element per Head)
                int token_stride = num_heads * d_head;

                // We transpose our store into K_tile, we store which element we are at within our K_cache
                K_tile[threadIdx.x][threadIdx.y] = K_cache[
                    phys_page     * page_stride  +  // (Which page) *  (How many Elements Per Page)
                    token_in_page * token_stride +  // (Which token) * (How many Elements per Token)
                    head_idx      * d_head       +  // (Which head) *  (How many Elements per Head)
                    k_d_idx                         // (Which element)
                ];
            } else {
                // Pad out of bounds, matching our transpose pattern
                K_tile[threadIdx.x][threadIdx.y] = 0.0f;
            }

            __syncthreads();

    /**************************** COMPUTE our Q * K DOT product (our similarity scores) ***************************************************** */
            for (int k = 0; k < TILE_SIZE; k++)
            // Store our reuslt in S_tile row wise
            //                                 Q_tile[fixed][iterate across columns] * K_tile[iterate down row][parallelize across columns]
                S_tile[threadIdx.y][threadIdx.x] += Q_tile[threadIdx.y][k] * K_tile[k][threadIdx.x];

            __syncthreads();
        }

        // Apply our constant scale to our values, ensures no numerical overflow
        S_tile[threadIdx.y][threadIdx.x] *= constant_scale;

        // Our column within matrix K
        //           (Which Tile * (Size of our Tile) + (Which column witin our Tile)
        int k_col_global = tileId_LongSide * TILE_SIZE + threadIdx.x;
        
        //Bounds check
        if (k_col_global >= seq_k)
            //Ensure our out of bounds does not effect our softmax values
            S_tile[threadIdx.y][threadIdx.x] = -INFINITY;

        __syncthreads();



    /**************************** COMPUTE our max and runningSum ***************************************************** */

        // Ensure our number will always over write our initialized tile_max
        // Note this is our max within our CURRENT tile (not across all tiles)
        float tile_max = -INFINITY;

        //Iterate through out our TILE
        for (int k = 0; k < TILE_SIZE; k++)
            // If our currentNum is above our tile_max, replace our tile_max
            //                           S_tile[current row, fixed][iterate through columns]
            //                                                      We use k to ensure we check across the entire tile
            //                                                      If we were to use threadIdx.x, we would find our own threads value within tile, not others
            tile_max = fmaxf(tile_max, S_tile[threadIdx.y][k]);

        // We compare our tile_max to our global max across all tiles thus far
        float new_max    = fmaxf(max_number, tile_max);

        // Gives how much we need to scale our old values to our new values, we assume we found a new max.
        // We could have a if statement, but that causes warp divergence, and increased instruction overhead.
        float correction = expf(max_number - new_max);

    //------------------- We scale our old values to our new values using A * B + C, where (oldValues) * (oldValues to NewValues bridge) + (newValue Scalar, our currentNum)
        //Update our running sum (A * B)
        runningSum *= correction;

        //Iterate throughout our tile
        for (int k = 0; k < TILE_SIZE; k++)
            // We add our currentNum to our runningSum, this is our + C to finish our bridge from our old scale to new scale
            // We subtract our currentNum by max to ensure no numerical overflow
            runningSum += expf(S_tile[threadIdx.y][k] - new_max);

        // Set our global max to our current local new max
        max_number = new_max;

        // Update our softmax_numerator (A * B), we must materialize V for our + C term
        softmax_numerator *= correction;

    /**************************** LOAD our V matrice into V_tile shared memory ***************************************************** */
        // Which row within our V matrice
        //          (Which Tile) * (Tile Size) + (Which row within tile)
        int v_row = tileId_LongSide * TILE_SIZE + threadIdx.y;
        // Our column within our V matrice
        int v_col = col;

        //Bounds check
        if (v_row < seq_k && v_col < d_head) {

            // Dimensional Analysis
            // Page -> Token -> Head -> Element
            //

            //Which row are we in our page (2D index)
            int logical_page  = v_row / PAGE_SIZE;
            // Which column are we in our row (2D index)
            int token_in_page = v_row % PAGE_SIZE;
            // Which page are we in (1D address)
            int phys_page     = this_page_table[logical_page];

            // How many Elements per Page
            //         (Tokens per Page) * (Heads per Token) * (Elements per Head)
            int page_stride  = PAGE_SIZE * num_heads * d_head;

            // How many Elements Per Token
            //         (Heads per Token) * (Elements Per Head)
            int token_stride = num_heads * d_head;

            // Add our current element to V_tile
            V_tile[threadIdx.y][threadIdx.x] = V_cache[
                phys_page     * page_stride  +  // (Which page)  * (Elements per Page)  + 
                token_in_page * token_stride +  // (Which Token) * (Elements per Token) +
                head_idx      * d_head       +  // (Which Head)  * (Elements per Head)  +
                v_col                           // Which element (scalar offset)
            ];
        } else {
            //Pad 0.0f for calculations later
            V_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

    /**************************** COMPUTE, + C for our softmax_numerator AND weight our similarity scores ***************************************************** */    
        // Iterate over our tile
        for (int k = 0; k < TILE_SIZE; k++)
            // Add our current value (+ C) to finish our A * B + C bridge from old scale to new scale AND give our similarity scores meaning by multiplying by our V_tile values, making it a WEIGHTED value
            //                  (S_tile([fixed][iterator]) - maxNum)   *    V_tile([iterator][parallelized])
            softmax_numerator += expf(S_tile[threadIdx.y][k] - max_number) * V_tile[k][threadIdx.x];

        __syncthreads();
    }

    /**************************** COMPUTE, normalize our weighted similarity scores  ***************************************************** */
    // Bounds check
    if (row < seq_q && col < d_head)
        // Normalize our weighted similarity score, output to O matrix
        O_head[row * d_head + col] = softmax_numerator / runningSum;

    // Bounds check
    // If our row is within our seq_q AND our col == 0
    // Any row, must be first column
    if (row < seq_q && col == 0) {
        // Which Token are we in within our Batch              
        //                 (Batch * Head) * (Tokens Per Head) + (Which token)
        long long lse_offset = (long long)blockIdx.z * seq_q + row;

        //Our token offset within our batch = log of runningSum + maxNumber
        //                                  = This is similar to runningSum * exp(max_num)
        out_lse[lse_offset] = logf(runningSum) + max_number;

        // For a scalar
        // In normal space, we do + and -
        // In exp space we do * and /
        // In log space(exp), we do + and - since they cancel out
    }
}























// BELOW CODE IS AI GENERATED


/**************************** Helpers ***************************************************** */

static void fill_rand(float* buf, int n) {
    for (int i = 0; i < n; i++) buf[i] = ((float)rand() / RAND_MAX) - 0.5f;
}

static float max_abs_error(const float* a, const float* b, int n) {
    float e = 0.f;
    for (int i = 0; i < n; i++) e = fmaxf(e, fabsf(a[i] - b[i]));
    return e;
}

// Rearranges contiguous K/V [batch, heads, seq_k, d_head]
// into paged layout    [total_pages, PAGE_SIZE, heads, d_head]
static void build_paged_cache(
    const float* src, float* dst,
    int batch_size, int num_heads, int seq_k, int d_head,
    int pages_per_batch
) {
    for (int b = 0; b < batch_size; b++)
    for (int h = 0; h < num_heads; h++)
    for (int t = 0; t < seq_k; t++) {
        int lp  = t / PAGE_SIZE;
        int tip = t % PAGE_SIZE;
        int pp  = b * pages_per_batch + lp;
        for (int d = 0; d < d_head; d++) {
            int from = b*(num_heads*seq_k*d_head) + h*(seq_k*d_head) + t*d_head + d;
            int to   = pp*(PAGE_SIZE*num_heads*d_head) + tip*(num_heads*d_head) + h*d_head + d;
            dst[to] = src[from];
        }
    }
}

// CPU reference: contiguous attention, used to verify paged kernel output
static void attention_cpu_ref(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int num_heads, int seq_q, int seq_k, int d_head, float scale
) {
    int hs_q  = seq_q * d_head, bs_q  = num_heads * hs_q;
    int hs_kv = seq_k * d_head, bs_kv = num_heads * hs_kv;
    float* S  = (float*)malloc(seq_q * seq_k * sizeof(float));

    for (int b = 0; b < batch_size; b++)
    for (int h = 0; h < num_heads; h++) {
        const float* Qh = Q + b*bs_q  + h*hs_q;
        const float* Kh = K + b*bs_kv + h*hs_kv;
        const float* Vh = V + b*bs_kv + h*hs_kv;
        float*       Oh = O + b*bs_q  + h*hs_q;

        for (int i = 0; i < seq_q; i++)
        for (int j = 0; j < seq_k; j++) {
            float s = 0.f;
            for (int k = 0; k < d_head; k++) s += Qh[i*d_head+k] * Kh[j*d_head+k];
            S[i*seq_k+j] = s * scale;
        }
        for (int i = 0; i < seq_q; i++) {
            float mx = -INFINITY, sum = 0.f;
            for (int j = 0; j < seq_k; j++) mx = fmaxf(mx, S[i*seq_k+j]);
            for (int j = 0; j < seq_k; j++) { S[i*seq_k+j] = expf(S[i*seq_k+j]-mx); sum += S[i*seq_k+j]; }
            for (int j = 0; j < seq_k; j++) S[i*seq_k+j] /= sum;
        }
        for (int i = 0; i < seq_q; i++)
        for (int d = 0; d < d_head; d++) {
            float s = 0.f;
            for (int j = 0; j < seq_k; j++) s += S[i*seq_k+j] * Vh[j*d_head+d];
            Oh[i*d_head+d] = s;
        }
    }
    free(S);
}

/**************************** Test runner ***************************************************** */

static void run_test(const char* label,
                     int batch_size, int num_heads,
                     int seq_q, int seq_k, int d_head) {
    float scale          = 1.0f / sqrtf((float)d_head);
    int pages_per_batch  = (seq_k + PAGE_SIZE - 1) / PAGE_SIZE;
    int total_pages      = batch_size * pages_per_batch;

    int total_q   = batch_size * num_heads * seq_q * d_head;
    int total_kv  = batch_size * num_heads * seq_k * d_head;
    int paged_kv  = total_pages * PAGE_SIZE * num_heads * d_head;

    float *h_Q   = (float*)malloc(total_q  * sizeof(float));
    float *h_K   = (float*)malloc(total_kv * sizeof(float));
    float *h_V   = (float*)malloc(total_kv * sizeof(float));
    float *h_ref = (float*)malloc(total_q  * sizeof(float));
    float *h_O   = (float*)malloc(total_q  * sizeof(float));

    float *h_K_paged = (float*)calloc(paged_kv, sizeof(float));
    float *h_V_paged = (float*)calloc(paged_kv, sizeof(float));
    int   *h_pt      = (int*)  malloc(batch_size * pages_per_batch * sizeof(int));

    fill_rand(h_Q, total_q);
    fill_rand(h_K, total_kv);
    fill_rand(h_V, total_kv);

    build_paged_cache(h_K, h_K_paged, batch_size, num_heads, seq_k, d_head, pages_per_batch);
    build_paged_cache(h_V, h_V_paged, batch_size, num_heads, seq_k, d_head, pages_per_batch);

    // Trivial page table: physical page == logical page index
    for (int i = 0; i < batch_size * pages_per_batch; i++) h_pt[i] = i;

    attention_cpu_ref(h_Q, h_K, h_V, h_ref,
                      batch_size, num_heads, seq_q, seq_k, d_head, scale);

    float *d_Q, *d_K_paged, *d_V_paged, *d_O, *d_lse;
    int   *d_pt;
    cudaMalloc(&d_Q,       total_q  * sizeof(float));
    cudaMalloc(&d_K_paged, paged_kv * sizeof(float));
    cudaMalloc(&d_V_paged, paged_kv * sizeof(float));
    cudaMalloc(&d_O,       total_q  * sizeof(float));
    cudaMalloc(&d_lse,     batch_size * num_heads * seq_q * sizeof(float));
    cudaMalloc(&d_pt,      batch_size * pages_per_batch * sizeof(int));

    cudaMemcpy(d_Q,       h_Q,       total_q  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K_paged, h_K_paged, paged_kv * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_paged, h_V_paged, paged_kv * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pt,      h_pt,      batch_size * pages_per_batch * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((d_head + TILE_SIZE-1)/TILE_SIZE,
                (seq_q  + TILE_SIZE-1)/TILE_SIZE,
                batch_size * num_heads);

    attention_paged_v1<<<blocks, threads>>>(
        d_Q, d_K_paged, d_V_paged, d_pt,
        d_O, d_lse,
        batch_size, num_heads, seq_q, seq_k, d_head,
        pages_per_batch, scale
    );
    cudaDeviceSynchronize();
    cudaMemcpy(h_O, d_O, total_q * sizeof(float), cudaMemcpyDeviceToHost);

    float err = max_abs_error(h_O, h_ref, total_q);
    const char* verdict = (err < 1e-3f) ? "PASS" : (err < 1e-2f) ? "FAIL (numerical)" : "FAIL";
    printf("  %-40s  max_err=%.2e  %s\n", label, err, verdict);

    free(h_Q); free(h_K); free(h_V); free(h_ref); free(h_O);
    free(h_K_paged); free(h_V_paged); free(h_pt);
    cudaFree(d_Q); cudaFree(d_K_paged); cudaFree(d_V_paged);
    cudaFree(d_O); cudaFree(d_lse); cudaFree(d_pt);
}

/**************************** Main ***************************************************** */

int main() {
    srand(42);
    printf("\n=== Week 5: Paged KV Attention (trivial page table) ===\n");
    printf("All tests compare paged kernel vs CPU reference with page_table[i]=i\n\n");

    run_test("Square  b=1 h=4  sq=64  sk=64  d=32",  1, 4,  64,  64, 32);
    run_test("Square  b=2 h=4  sq=48  sk=48  d=32",  2, 4,  48,  48, 32);
    run_test("sk>sq   b=1 h=4  sq=64  sk=128 d=32",  1, 4,  64, 128, 32);
    run_test("sk>sq   b=2 h=8  sq=96  sk=192 d=64",  2, 8,  96, 192, 64);
    run_test("sq>sk   b=1 h=4  sq=128 sk=64  d=32",  1, 4, 128,  64, 32);
    run_test("sk=PAGE b=1 h=4  sq=32  sk=64  d=32",  1, 4,  32,  64, 32);
    run_test("sk=2xPG b=1 h=4  sq=32  sk=128 d=32",  1, 4,  32, 128, 32);
    run_test("sk odd  b=1 h=4  sq=32  sk=80  d=32",  1, 4,  32,  80, 32);

    return 0;
}

/**************************** PyTorch extension ***************************************************** */

#ifdef WITH_TORCH
#include <torch/extension.h>

torch::Tensor paged_attention_forward(
    torch::Tensor Q,
    torch::Tensor K_cache,
    torch::Tensor V_cache,
    torch::Tensor page_table,
    int seq_k,
    float scale
) {
    int batch_size        = Q.size(0);
    int num_heads         = Q.size(1);
    int seq_q             = Q.size(2);
    int d_head            = Q.size(3);
    int max_logical_pages = page_table.size(1);

    torch::Tensor O   = torch::empty_like(Q);
    torch::Tensor lse = torch::empty({batch_size * num_heads, seq_q},
                                     torch::dtype(torch::kFloat32).device(Q.device()));

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((d_head + TILE_SIZE-1)/TILE_SIZE,
                (seq_q  + TILE_SIZE-1)/TILE_SIZE,
                batch_size * num_heads);

    attention_paged_v1<<<blocks, threads>>>(
        Q.data_ptr<float>(),
        K_cache.data_ptr<float>(),
        V_cache.data_ptr<float>(),
        page_table.data_ptr<int>(),
        O.data_ptr<float>(),
        lse.data_ptr<float>(),
        batch_size, num_heads, seq_q, seq_k, d_head,
        max_logical_pages, scale
    );
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &paged_attention_forward, "paged attention v1");
}
#endif