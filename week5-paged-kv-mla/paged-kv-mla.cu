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

















/*
 DeepSeek-V2 MLA Attention

 After reading: 
 - DeepSeek's V2 paper
    > Specifically Section 2.1 (Multi-Head Latent Attention)
    > Specifically Section 3.1.2 (Hyper-Parameters)
 - FlashInfer Bench
    > Specifically dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64

I have the following mental model of Multi-Head Latent Attention for this competition

- My understanding of the DeepSeek paper proposes the following structure:
   > W^DKV contains our down projection of our KV (C^KV)
   > C^KV contains our compressed K and V. Just the contents, not our rotational positional vectors.
   > C^KV up projects to both K_nope (K contents) and V (V contents)
   > K_nope (K contents) concatenated with K_pe (K RoPE) for our original K with contents/positions
   > With our (Q_nope ; Q_pe) DOT (K_nope ; K_pe) * V, we have our Q K V attention

- Our kernel for this competition does the following:
    > Given:
       - q_nope                                 q_nope -> BFLOAT16 -> Q CONTENTS
          > num_tokens   (X) (var)
          > num_qo_heads (Y) (16)
          > head_dim_ckv (Z) (512)
       - q_pe                                   q_pe   -> BFLOAT16 -> Q POSITION
          > num_tokens   (X) (var)
          > num_qo_heads (Y) (16)
          > head_dim_kpe (Z) (64)
       - ckv_cache                           ckv_cache -> BFLOAT16 -> K & V CONTENTS
          > num_pages    (X) (var)
          > page_size    (Y) (64)
          > head_dim_ckv (Z) (512)
       - kpe_cache                           kpe_cache -> BFLOAT16 -> K POSITION
          > num_pages    (X) (var)
          > page_size    (Y) (64)
          > head_dim_kpe (Z) (64)
       - sparse_indices                 sparse_indices ->  INT32   -> SPARSE MASK !! NOT YET INCLUDED !!
          > num_tokens   (X) (var)
          > topk         (Y) (2048)
       - sm_scale                             sm_scale ->  FLOAT32 -> SCALAR CONSTANT
          > scalar       (X) (var)






    In our memory hierarchy we organize by
     - Biggest -> Medium -> Smallest
         (X)        (Y)        (Z)
     -  Token   -> Heads  -> Element

    So that when we access it, our smallest unit (element) is contingous


    In our execution hierarchy, we organize by
     - Smallest -> Medium -> Biggest
         (X)        (Y)        (Z)

    Our smallest (threadIdx.x) is in lockstep and parallelized

    
    This means when we map our execution hierarchy to our memory hierarchy, we have the following mapping
        Execution           Memory
           (X)      ->       (Z)       -> X is parallelized, and Z is contingous/coalesced

    This tells us to maximize our threadIdx.x parallelization, we should process our Z dimension, 
    which has contingous access between elements (stored right next to each other on hardware)



    DIMENSIONAL ANALYSIS TABLE:
     - q_nope, q_pe: 
        Token -> Head -> Element

     - ckv_cache, kpe_cache
        Pages -> Tokens -> Element

We DOT product our q_nope and ckv_cache, we know this by their matching Z dimension (dimension to be reduced), and q_nope provides our Q contents, while ckv_cache provides our K contents
    RESULT: This gives us our un-weighted/un-normalized similarity scores
        - NOTE: ckv_cache not only provides us our K contents, but also our V contents during our weighted output accumulation
We DOT product our q_pe and kpe_cache, we know this by their matching Z dimension (dimension to be reduced), q_pe provides our Q position, while kpe_cache provides our K position
    RESULT: This gives us our Q and K positional similarity scores


IMPORTANT:
 The difference between the papers implementation, and our FlashInfer MLSys 2026 implementation is the following:
   - The paper describes for us to calculate the DOT product of our Q * K, we must do the following:
      > q * (W_UK * ckv_cache)
   - But our kernel's DOT product of Q * K, we only need to do
      > q * ckv_cache
 The reason for this is our provided q_nope (the contents of our Q), already was multiplied by W_UK before the kernel. 
 Essentially because multiplication is associative (grouping does not matter), we can get away with
   - (q * W_UK) * ckv_cache


------------------------
 After completing the kernel, the most helpful thing was to draw our input tensors, and our output tensor, and then figuring out how to use our
 input tensors to reach our output tensor while following the paper's procedure. This helped me understand When we do 3D tensor DOT product 3D tensor,
 we get a 4D tensor, and our Z dimension (the common dimension between both tensors) gets reduced. 

 When we dot product:
 - ckv_cache with q_nope we get a 4D tensor of [num_pages, page_size, num_tokens, num_qo_heads] (CONTENT)
 - kpe_cache with q_ope we get a 4D tensor of [num_pages, page_size, num_tokens, num_qo_heads]  (POSITION)

 Since both of these DOT products give us the same tensor layout, we can add them to each other.

 Looking at our output, we know we need a 3D tensor of [num_tokens, num_qo_heads, head_dim_ckv] this tells us we are not done (also by reading our paper we know we are not done and have one more step)
 To turn 4D tensor into a 3D tensor, we can re-use our ckv_cache tensor where:
 - Our 4D Tensor is [num_pages, page_size, num_tokens, num_qo_heads]
 - Our 3D Tensor is [num_pages, page_size, head_dim_ckv]

 When we DOT product these two tensors, their matching dimensions get reduced/summed over. These matching dimensions are

 - [num_pages] and [page_size]

 And all of our remaining dimensions are our output dimensions. This leaves us with a final 3D tensor of 

 - [num_tokens, num_qo_heads, head_dim_ckv]

 This tells us the DOT products and additions we are doing. Note, geometry tells us how the data is shaped, and what operations are needed to be done.
 But they do not tell us the order of operations, nor do they tell us any variables to track or the control flow of our kernel such as Load, Compute, Store etc.

*/







__global__ void attention_mla_v1(
    const float* __restrict__ q_nope,
    const float* __restrict__ q_pe,
    const float* __restrict__ ckv_cache,
    const float* __restrict__ kpe_cache,
    const int*   __restrict__ page_table,
    float*       __restrict__ O,
    float*       __restrict__ out_lse,
    int num_tokens,
    int num_qo_heads,
    int num_kv_tokens,
    int head_dim_ckv,
    int head_dim_kpe,
    int max_logical_pages,
    float sm_scale
) {
/**************************** Initialize row pointers & Shared memory***************************************************** */
    //Our shared memory tiles, we traverse Qpe_tile and Kpe_tile column wise, giving us an implicit transpose, hence we must pad our column by one to ensure no bank conflicts
    __shared__ float Q_tile  [TILE_SIZE][TILE_SIZE];
    __shared__ float K_tile  [TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Qpe_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float Kpe_tile[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float S_tile  [TILE_SIZE][TILE_SIZE];
    __shared__ float V_tile  [TILE_SIZE][TILE_SIZE];

    // Our current head (Z)
    int head_idx = blockIdx.z;

    // Our current thread row within our tile
    //       (CURRENT BLOCK ROW) * (SIZE OF TILE) + (WHICH THREAD ROW WITHIN TILE)
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;

    //Our current thread column within our tile
    //       (CURRENT BLOCK COLUMN) * (SIZE OF TILE) + (WHICH THREAD COLUMN WITHIN TILE)
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Stride to go from token to token, CONTENT TENSOR
    //               (HOW MANY HEADS) * (HOW MANY ELEMENTS) = HOW MANY ELEMENTS ACROSS ALL HEADS IN A SINGULAR TOKEN
    int token_stride_q = num_qo_heads * head_dim_ckv;

    // Stride to go from token to token, POSTIONAL TENSOR
    //                 (HOW MANY HEADS) * (HOW MANY ELEMENTS) = HOW MANY ELEMENTS ACROSS ALL HEADS IN A SINGULAR TOKEN
    int token_stride_qpe = num_qo_heads * head_dim_kpe;

    // Determine which element we are at within our current head, CONTENT TENSOR
    //                     (WHICH HEAD, COLUMN) * (HOW MANY ELEMENTS IN HEAD) + (WHICH ELEMENT WITHIN HEAD)
    const float* Q_nope_head = head_idx * head_dim_ckv + q_nope;

    // Determine which element we are at within our current head, POSTIONAL TENSOR
    //                     (WHICH HEAD, COLUMN) * (HOW MANY ELEMENTS IN HEAD) + (WHICH ELEMENT WITHIN HEAD)
    const float* Q_pe_head = head_idx * head_dim_kpe + q_pe;

    // Determine which element we are at within our current head, OUTPUT TENSOR
    //           (WHICH HEAD, COLUMN) * (HOW MANY ELEMENTS IN HEAD) + (WHICH ELEMENT WITHIN HEAD)
    float* O_head = head_idx * head_dim_ckv + O;

    float max_val = -INFINITY;
    float running_sum = 0.0f;
    float output_acc = 0.0f;

    // How many times does our TILE need to SLIDE across our num_kv_tokens dimension to process the entire row
    int num_kv_tiles = (num_kv_tokens + TILE_SIZE - 1) / TILE_SIZE;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {

/**************************** LOAD our q_nope_head elements into Q_tile shared memory >> Q_NOPE, EXECUTION HIERARCHY, CONTENT TENSOR***************************************************** */
        //Intialize our S_tile to later be added upon. 0.0f initialization to ensure proper addition later on
        S_tile[threadIdx.y][threadIdx.x] = 0.0f;

        // How many times does our TILE needs iterate to COVER our TILE's current position in the head_dim_ckv dimension in BOTH ckv_cache and q_nope tensors.
        // This dimension will be reduced and summed over.
        int num_ckv_tiles = (head_dim_ckv + TILE_SIZE - 1) / TILE_SIZE;

        for (int d_tile = 0; d_tile < num_ckv_tiles; d_tile++) {

            // Our current column location within our tile 
            //       (CURRENT TILE LOCATION) * (TILE_SIZE) + (CURRENT THREAD COLUMN WITHIN TILE)
            int q_col = d_tile * TILE_SIZE + threadIdx.x;

            //Bounds check
            if (row < num_tokens && q_col < head_dim_ckv)
                // Add our current ELEMENT COLUMN within our current q_nope_head to Q_tile (not transposed)
                //                                           (CURRENT ROW) * (ROW SIZE) + (CURRENT COLUMN WITHIN ROW)
                Q_tile[threadIdx.y][threadIdx.x] = Q_nope_head[row * token_stride_q + q_col];
            else
                // Pad 0.0f if out of bounds for later easy calculations
                Q_tile[threadIdx.y][threadIdx.x] = 0.0f;


/**************************** LOAD our K elements from our paged KV cache into K_tile shared memory >> CKV_CACHE, MEMORY HIERARCHY, CONTENT TENSOR ***************************************************** */
            // Our current thread (column-wise) in our head_dim_ckv dimension of our Q_NOPE, EXECUTION HIERARCHY, CONTENT TENSOR
            int k_d_idx = d_tile * TILE_SIZE + threadIdx.x;
            // Our current thread (row-wise) in our num_kv_tokens dimension of our Q_NOPE, EXECUTION HIERARCHY, CONTENT TENSOR
            int k_seq_idx = kv_tile * TILE_SIZE + threadIdx.y;

            //Bounds check 
            if (k_seq_idx < num_kv_tokens && k_d_idx < head_dim_ckv) {
                // Our bridge between our execution hierarchy tensor q_nope and our memory hierarchy tensor ckv_cache. Both are content tensors
                
                // CONVERT OUR 1D HARDWARE ADDRESS (K_SEQ_IDX) INTO A LOGICAL 2D INDEX
                // Which row does our column thread from q_nope map onto our ckv_cache tensor
                int logical_page = k_seq_idx / PAGE_SIZE;
                //Which column does our row thread from q_nope map onto our ckv_cache tensor
                int token_in_page = k_seq_idx % PAGE_SIZE;

                // Return respective page indice given our page row index
                int phys_page = page_table[logical_page];

                // Stride to go page to page (X dimension of our ckv_tensor)
                //                  (HOW MANY TOKENS PER PAGE) * (HOW MANY ELEMENTS PER TOKEN)
                int page_stride_ckv = PAGE_SIZE * head_dim_ckv;

                // Stride to go token to token (Y dimension of our ckv_tensor)
                int token_stride_ckv = head_dim_ckv;

                // Store our current element, implicitly transposed in shared memory for later DOT product
                K_tile[threadIdx.x][threadIdx.y] = ckv_cache[
                    phys_page     * page_stride_ckv +  // (CURRENT PAGE, X)  * (SIZE OF PAGE, Y) + 
                    token_in_page * token_stride_ckv + // (CURRENT TOKEN, Y) * (SIZE OF TOKEN, Z) + 
                    k_d_idx                            // (CURRENT ELEMENT, Z)
                ];
            } else {
                //Pad for easy computation later
                K_tile[threadIdx.x][threadIdx.y] = 0.0f;
            }

            __syncthreads();
/**************************** COMPUTE similiarity between vectors in our ckv_cache tensor and our q_nope tensor, reducing head_dim_ckv dimension***************************************************** */
            for (int k = 0; k < TILE_SIZE; k++)
                S_tile[threadIdx.y][threadIdx.x] += Q_tile[threadIdx.y][k] * K_tile[k][threadIdx.x];

            __syncthreads();
        }

/**************************** LOAD our q_pe head elements into Q_tile shared memory >> Q_PE, EXECUTION HIERARCHY, POSITIONAL TENSOR***************************************************** */
        // How many times does our TILE need to slide across the head_dim_kpe dimension in BOTH of our kpe_cache and q_pe tensors
        int num_kpe_tiles = (head_dim_kpe + TILE_SIZE - 1) / TILE_SIZE;

        for (int d_tile = 0; d_tile < num_kpe_tiles; d_tile++) {

            // Our current thread within the tile
            //           (CURRENT TILE * (TILE SIZE) + (WHICH COLUMN IN TILE)
            int qpe_col = d_tile * TILE_SIZE + threadIdx.x;

            //Bounds check
            if (row < num_tokens && qpe_col < head_dim_kpe)
                //Store within our Qpe_tile
                //                 Within our Q_pe_heads (Y dimension) [Which head (Y) * How many tokens per head (X) + Which token within our head(Z)]
                Qpe_tile[threadIdx.y][threadIdx.x] = Q_pe_head[row * token_stride_qpe + qpe_col];
            else
                //Pad for later calculation
                Qpe_tile[threadIdx.y][threadIdx.x] = 0.0f;

            // Bridge our current thread column location in our Q_PE POSITIONAL TENSOR (EXECUTION HIERARCHY) to our KPE_CACHE POSITIONAL TENSOR (MEMORY HIERARCHY)
            //             (CURRENT TILE) * (SIZE OF TILE) + (CURRENT COLUMN WITHIN TILE)
            int kpe_d_idx = d_tile * TILE_SIZE + threadIdx.x;
            // Bridge our current thread row location in our Q_PE POSITIONAL TENSOR (EXECUTION HIERARCHY) to our KPE_CACHE POSITIONAL TENSOR (MEMORY HIERARCHY)
            //             (CURRENT TILE) * (SIZE OF TILE) + (CURRENT ROW WITHIN TILE)
            int kpe_seq_idx = kv_tile * TILE_SIZE + threadIdx.y;

            //Bounds check as we traverse both tensors
            if (kpe_seq_idx < num_kv_tokens && kpe_d_idx < head_dim_kpe) {
                //Convert our 1D hardware address to a 2D logical address
                // Our current row within our kpe_cache positional tensor, memory hierarchy
                int logical_page = kpe_seq_idx / PAGE_SIZE;
                // Our current column within our kpe_cache positional tensor, memory hierarchy
                int token_in_page = kpe_seq_idx % PAGE_SIZE;

                // Our physical index of where our elements are stored within our paged memory hierachy
                int phys_page = page_table[logical_page];

                // Stride between the current page and the next
                //   (HOW MANY TOKENS PER PAGE) * (HOW MANY ELEMENTS PER TOKEN)
                int page_stride_kpe = PAGE_SIZE * head_dim_kpe;

                // Stride between our current token and the next
                //                      (HOW MANY ELEMENTS PER TOKEN)
                int token_stride_kpe = head_dim_kpe;

                // Store our current element within our Kpe_tile, stored transposed for later DOT product. Must pad Kpe_tile column by one.
                Kpe_tile[threadIdx.x][threadIdx.y] = kpe_cache[
                    phys_page     * page_stride_kpe +   // (CURRENT PAGE)  * (SIZE OF PAGE)
                    token_in_page * token_stride_kpe +  // (CURRENT TOKEN) * (SIZE OF TOKEN)
                    kpe_d_idx                           // (OUR CURRENT ELEMENT)
                ];
            } else {
                //Pad for later easy computation
                Kpe_tile[threadIdx.x][threadIdx.y] = 0.0f;
            }

            __syncthreads();

/**************************** COMPUTE similiarity between vectors in our kpe_cache tensor and our q_pe tensor, reducing head_dim_kpe (elements) dimension***************************************************** */
            for (int k = 0; k < TILE_SIZE; k++)
                // Add to our positional similarity to S_tile values, which already has our scores for Q and K's CONTENT similarity, our output is a 4D Tensor stored by S_Tile
                // our output is a 4D Tensor stored by S_Tile
                //        Qpe_tile[current row, fixed][iterate over columns] * Kpe_tile[iterate down row][parallelize across column]
                S_tile[threadIdx.y][threadIdx.x] += Qpe_tile[threadIdx.y][k] * Kpe_tile[k][threadIdx.x];

            __syncthreads();
        }

/**************************** COMPUTE Apply a constant to S_tile to avoid numerical overflow***************************************************** */

        // Apply a constant scale to all of our values within S_tile to avoid numerical overflow
        S_tile[threadIdx.y][threadIdx.x] *= sm_scale;

        // Our current column within our tile
        //              (TILE LOCATION) * (TILE SIZE) + (CURRENT COLUMN WITHIN TILE)
        int k_col_global = kv_tile * TILE_SIZE + threadIdx.x;

        //Bounds check, ensure any column without a valid token does not effect our softmax
        if (k_col_global >= num_kv_tokens)
            S_tile[threadIdx.y][threadIdx.x] = -INFINITY;

        __syncthreads();

/**************************** COMPUTE Calculate/scale our softmax runningSum using the universal A * B + C brdige***************************************************** */
        // Our current tile's maximum
        float tile_max = -INFINITY;
        
        //Iterate across tile
        for (int k = 0; k < TILE_SIZE; k++)
            //Check if currentNum is above local tile_max
            tile_max = fmaxf(tile_max, S_tile[threadIdx.y][k]);

        // Compare tile_max to max_val across all tiles
        float new_max = fmaxf(max_val, tile_max);
        
        //Calculate B in the bridge formula A * B + C. This allows us to scale old values to newer values when we come across a new max.
        // A key insight that softmax uses to only materalize our matrices once.
        float correction = expf(max_val - new_max);

        // Apply our A * B bridge to our running_sum
        running_sum *= correction;

        //Iterate across our tile
        for (int k = 0; k < TILE_SIZE; k++)
            // Add our + C, the currentNumber scaled to the new maximum to finish our A * B + C bridge
            running_sum += expf(S_tile[threadIdx.y][k] - new_max);

        // Set our local tile max, to the max we've seen across all tiles thus far
        max_val = new_max;

/**************************** COMPUTE Calculate/scale our softmax numerator using A * B + C***************************************************** */
        // Our A * B bridge, our + C term requires us to materalize our V_tile
        output_acc *= correction;

/**************************** LOAD our V_tile to later calculate our + C***************************************************** */
        // Our current row within V tile, whom iterates across the ckv_cache, specifically the page_size(Y) dimension which contains how many tokens per page
        //   (CURRENT TILE) * (TILE_SIZE) + (threadIdx.y)
        int v_row = kv_tile * TILE_SIZE + threadIdx.y;
        // Our current column within our blockIdx.x (execution hierarchy) 
        int v_col = col;

        //Bounds check 
        if (v_row < num_kv_tokens && v_col < head_dim_ckv) {
            //Map our 1D hardware address to a 2D logical index
            // Our current row within our pages
            int logical_page = v_row / PAGE_SIZE;
            // Our current column within our page 
            int token_in_page = v_row % PAGE_SIZE;

            // Our page index of where we are 
            int phys_page = page_table[logical_page];

            // Stride between the current page and the next
            //   (HOW MANY TOKENS PER PAGE) * (HOW MANY ELEMENTS PER TOKEN)
            int page_stride_ckv = PAGE_SIZE * head_dim_ckv;

            // Stride between our current token and the next
            //               (HOW MANY ELEMENTS PER TOKEN)
            int token_stride_ckv = head_dim_ckv;

            // Store our current element within our V_tile
            V_tile[threadIdx.y][threadIdx.x] = ckv_cache[
                phys_page     * page_stride_ckv +   // (CURRENT PAGE) * (SIZE OF PAGE, giving us the unit [TOKENS])
                token_in_page * token_stride_ckv +  // (CURRENT TOKEN) * (SIZE OF TOKEN, giving us the unit [ELEMENT])
                v_col                               // (CURRENT ELEMENT)
            ];
        } else {
            //Pad for later computation
            V_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

/**************************** COMPUTE our + C term for our softmax numerator**************************************************** */
        //Iterate over our tile
        for (int k = 0; k < TILE_SIZE; k++)
            // DOT product of our S_Tile (contains CONTENT and POSITION) by our V_tile (CONTAINS MEANING FOR OUR NUMBERS)
            // We reduce/sum over our 4D S_tile tensor into a 3D tensor matching our output of num_tokens (X), num_qo_heads(Y) and head_dim_ckv(Z)
            //             S_tile(currentNum - maxNum) * V_tile()             
            //
            //             S_tile([fixed row][iterate over columns]) * V_tile([iterate down rows][parallelize over columns])
            output_acc += expf(S_tile[threadIdx.y][k] - max_val) * V_tile[k][threadIdx.x];

        __syncthreads();
    }

/**************************** COMPUTE and STORE our weighted and normalized similarity value**************************************************** */
    //Bounds check
    if (row < num_tokens && col < head_dim_ckv)
        // Store our weighted and normalized similarity value into our output tensor
        //  (CURRENT ROW, Y) * (SIZE OF ROW, X) + (WHICH COLUMN IN ROW, X) = currentNum / runningSum
        O_head[row * token_stride_q + col] = output_acc / running_sum;

/**************************** STORE our runningSum for our next kernel launch **************************************************** */
    //Bounds check, each of our respective tokens stores their runningSum (made up of heads)
    if (row < num_tokens && col == 0) {
        //Offset between 
        //                     (CURRENT TOKEN) * (HEADS PER TOKEN) + (CURRENT HEAD WITHIN TOKEN)
        long long lse_offset = (long long)row * num_qo_heads + head_idx;

        //When we input a token, we want its respective heads value
        out_lse[lse_offset] = logf(running_sum) + max_val;
    }
}







































// BELOW CODE IS AI GENERATED FOR TESTS



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

/**************************** MLA Helpers ***************************************************** */

static void build_paged_cache_mla(
    const float* src, float* dst,
    int num_kv_tokens, int feat_dim,
    int pages_per_seq
) {
    for (int t = 0; t < num_kv_tokens; t++) {
        int lp = t / PAGE_SIZE;
        int tip = t % PAGE_SIZE;
        for (int d = 0; d < feat_dim; d++) {
            int from = t * feat_dim + d;
            int to = lp * (PAGE_SIZE * feat_dim) + tip * feat_dim + d;
            dst[to] = src[from];
        }
    }
}

static void attention_mla_cpu_ref(
    const float* q_nope, const float* q_pe,
    const float* ckv,    const float* kpe,
    float* O,
    int num_tokens, int num_heads,
    int num_kv_tokens,
    int head_dim_ckv, int head_dim_kpe,
    float scale
) {
    float* S = (float*)malloc(num_tokens * num_kv_tokens * sizeof(float));
    int stride_qn = num_heads * head_dim_ckv;
    int stride_qp = num_heads * head_dim_kpe;

    for (int h = 0; h < num_heads; h++) {
        const float* Qn = q_nope + h * head_dim_ckv;
        const float* Qp = q_pe + h * head_dim_kpe;
        float* Oh = O + h * head_dim_ckv;

        for (int i = 0; i < num_tokens; i++)
        for (int j = 0; j < num_kv_tokens; j++) {
            float s = 0.f;
            for (int d = 0; d < head_dim_ckv; d++)
                s += Qn[i * stride_qn + d] * ckv[j * head_dim_ckv + d];
            for (int d = 0; d < head_dim_kpe; d++)
                s += Qp[i * stride_qp + d] * kpe[j * head_dim_kpe + d];
            S[i * num_kv_tokens + j] = s * scale;
        }

        for (int i = 0; i < num_tokens; i++) {
            float mx = -INFINITY, sum = 0.f;
            for (int j = 0; j < num_kv_tokens; j++) mx = fmaxf(mx, S[i * num_kv_tokens + j]);
            for (int j = 0; j < num_kv_tokens; j++) { S[i * num_kv_tokens + j] = expf(S[i * num_kv_tokens + j] - mx); sum += S[i * num_kv_tokens + j]; }
            for (int j = 0; j < num_kv_tokens; j++) S[i * num_kv_tokens + j] /= sum;
        }

        for (int i = 0; i < num_tokens; i++)
        for (int d = 0; d < head_dim_ckv; d++) {
            float s = 0.f;
            for (int j = 0; j < num_kv_tokens; j++)
                s += S[i * num_kv_tokens + j] * ckv[j * head_dim_ckv + d];
            Oh[i * stride_qn + d] = s;
        }
    }
    free(S);
}

static void run_test_mla(const char* label,
                         int num_tokens, int num_heads,
                         int num_kv_tokens,
                         int head_dim_ckv, int head_dim_kpe) {
    float scale = 1.0f / sqrtf((float)(head_dim_ckv + head_dim_kpe));
    int pages_per_seq = (num_kv_tokens + PAGE_SIZE - 1) / PAGE_SIZE;

    int total_qn = num_tokens * num_heads * head_dim_ckv;
    int total_qp = num_tokens * num_heads * head_dim_kpe;
    int total_ckv = num_kv_tokens * head_dim_ckv;
    int total_kpe = num_kv_tokens * head_dim_kpe;
    int total_O = num_tokens * num_heads * head_dim_ckv;
    int paged_ckv = pages_per_seq * PAGE_SIZE * head_dim_ckv;
    int paged_kpe = pages_per_seq * PAGE_SIZE * head_dim_kpe;

    float *h_qn = (float*)malloc(total_qn * sizeof(float));
    float *h_qp = (float*)malloc(total_qp * sizeof(float));
    float *h_ckv = (float*)malloc(total_ckv * sizeof(float));
    float *h_kpe = (float*)malloc(total_kpe * sizeof(float));
    float *h_ref = (float*)malloc(total_O * sizeof(float));
    float *h_O = (float*)malloc(total_O * sizeof(float));
    float *h_ckv_paged = (float*)calloc(paged_ckv, sizeof(float));
    float *h_kpe_paged = (float*)calloc(paged_kpe, sizeof(float));
    int *h_pt = (int*)malloc(pages_per_seq * sizeof(int));

    fill_rand(h_qn, total_qn);
    fill_rand(h_qp, total_qp);
    fill_rand(h_ckv, total_ckv);
    fill_rand(h_kpe, total_kpe);

    build_paged_cache_mla(h_ckv, h_ckv_paged, num_kv_tokens, head_dim_ckv, pages_per_seq);
    build_paged_cache_mla(h_kpe, h_kpe_paged, num_kv_tokens, head_dim_kpe, pages_per_seq);

    for (int i = 0; i < pages_per_seq; i++) h_pt[i] = i;

    attention_mla_cpu_ref(
        h_qn, h_qp, h_ckv, h_kpe, h_ref,
        num_tokens, num_heads, num_kv_tokens,
        head_dim_ckv, head_dim_kpe, scale
    );

    float *d_qn, *d_qp, *d_ckv, *d_kpe, *d_O, *d_lse;
    int *d_pt;

    cudaMalloc(&d_qn, total_qn * sizeof(float));
    cudaMalloc(&d_qp, total_qp * sizeof(float));
    cudaMalloc(&d_ckv, paged_ckv * sizeof(float));
    cudaMalloc(&d_kpe, paged_kpe * sizeof(float));
    cudaMalloc(&d_O, total_O * sizeof(float));
    cudaMalloc(&d_lse, num_heads * num_tokens * sizeof(float));
    cudaMalloc(&d_pt, pages_per_seq * sizeof(int));

    cudaMemcpy(d_qn, h_qn, total_qn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qp, h_qp, total_qp * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ckv, h_ckv_paged, paged_ckv * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kpe, h_kpe_paged, paged_kpe * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pt, h_pt, pages_per_seq * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((head_dim_ckv + TILE_SIZE - 1) / TILE_SIZE,
                (num_tokens   + TILE_SIZE - 1) / TILE_SIZE,
                num_heads);

    attention_mla_v1<<<blocks, threads>>>(
        d_qn, d_qp, d_ckv, d_kpe, d_pt,
        d_O, d_lse,
        num_tokens, num_heads, num_kv_tokens,
        head_dim_ckv, head_dim_kpe,
        pages_per_seq, scale
    );

    cudaDeviceSynchronize();
    cudaMemcpy(h_O, d_O, total_O * sizeof(float), cudaMemcpyDeviceToHost);

    float err = max_abs_error(h_O, h_ref, total_O);
    const char* verdict = (err < 1e-3f) ? "PASS" : (err < 1e-2f) ? "FAIL (numerical)" : "FAIL";
    printf("  %-50s  max_err=%.2e  %s\n", label, err, verdict);

    free(h_qn); free(h_qp); free(h_ckv); free(h_kpe);
    free(h_ref); free(h_O); free(h_ckv_paged); free(h_kpe_paged); free(h_pt);
    cudaFree(d_qn); cudaFree(d_qp); cudaFree(d_ckv); cudaFree(d_kpe);
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

    printf("\n=== Week 5: MLA Paged Attention ===\n");
    printf("All tests compare MLA kernel vs CPU reference with page_table[i]=i\n\n");

    run_test_mla("square  tok=32  kv=64   h=4  ckv=32  kpe=16",  32,  4,  64,  32, 16);
    run_test_mla("square  tok=48  kv=48   h=4  ckv=32  kpe=16",  48,  4,  48,  32, 16);
    run_test_mla("kv>tok  tok=64  kv=128  h=4  ckv=32  kpe=16",  64,  4, 128,  32, 16);
    run_test_mla("kv>tok  tok=96  kv=192  h=8  ckv=64  kpe=32",  96,  8, 192,  64, 32);
    run_test_mla("tok>kv  tok=128 kv=64   h=4  ckv=32  kpe=16", 128,  4,  64,  32, 16);
    run_test_mla("kv=PAGE tok=32  kv=64   h=4  ckv=32  kpe=16",  32,  4,  64,  32, 16);
    run_test_mla("kv=2xPG tok=32  kv=128  h=4  ckv=32  kpe=16",  32,  4, 128,  32, 16);
    run_test_mla("kv odd  tok=32  kv=80   h=4  ckv=32  kpe=16",  32,  4,  80,  32, 16);
    run_test_mla("spec    tok=32  kv=64   h=16 ckv=512 kpe=64",  32, 16,  64, 512, 64);
    run_test_mla("spec    tok=64  kv=128  h=16 ckv=512 kpe=64",  64, 16, 128, 512, 64);

    return 0;
}

/**************************** PyTorch extension ***************************************************** */

#ifdef WITH_TORCH
#include <torch/extension.h>

std::vector<torch::Tensor> paged_attention_forward(
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
    return {O, lse};
}

std::vector<torch::Tensor> mla_attention_forward(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor page_table,
    int num_kv_tokens,
    float sm_scale
) {
    int num_tokens       = q_nope.size(0);
    int num_qo_heads     = q_nope.size(1);
    int head_dim_ckv     = q_nope.size(2);
    int head_dim_kpe     = q_pe.size(2);
    int max_logical_pages = page_table.size(0);

    torch::Tensor O   = torch::empty_like(q_nope);
    torch::Tensor lse = torch::empty({num_tokens, num_qo_heads},
                                     torch::dtype(torch::kFloat32).device(q_nope.device()));

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((head_dim_ckv + TILE_SIZE - 1) / TILE_SIZE,
                (num_tokens   + TILE_SIZE - 1) / TILE_SIZE,
                num_qo_heads);

    attention_mla_v1<<<blocks, threads>>>(
        q_nope.data_ptr<float>(),
        q_pe.data_ptr<float>(),
        ckv_cache.data_ptr<float>(),
        kpe_cache.data_ptr<float>(),
        page_table.data_ptr<int>(),
        O.data_ptr<float>(),
        lse.data_ptr<float>(),
        num_tokens, num_qo_heads, num_kv_tokens,
        head_dim_ckv, head_dim_kpe,
        max_logical_pages, sm_scale
    );
    return {O, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",     &paged_attention_forward, "paged attention v1");
    m.def("forward_mla", &mla_attention_forward,   "MLA paged attention v1");
}
#endif