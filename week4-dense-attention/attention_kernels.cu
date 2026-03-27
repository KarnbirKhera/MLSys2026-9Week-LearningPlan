
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*--------------- PRELUDE -----------------------*/
/*
My current understanding of attention is as follows:
 - We have three matrices:
    > Q represents our queries, we can see them ask questions
    > K represents our keys, our answer to said questions
    > V represents the actual values behind our keys. We use this later when we want to update
      our respective vector based on its similarity probability to the query vector.

 - We'll be using GEMM to perform our DOT product of Q * K, and our softmax to give us our "how much do I matter" distribution from 0 to 1, 
   to later be interpreted as 0-100% percentages.



*/

#define TILE_SIZE 32

__global__ void qkt_tiled(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float*       __restrict__ S,
    int seq_q,
    int seq_k,
    int numOfCols
) {
    /**************************** Initialize row pointers & Shared memory ***************************************************** */

    //If our threadIdx.x (or iterator) is our first column while we use our shared memory,
    // it must be padded to ensure each thread hits its own bank. We essentially make our modulus for shared memory from 32 -> 33
    __shared__ float Q_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float K_tile[TILE_SIZE][TILE_SIZE+1];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float softmax_numerator = 0.0f;
    int num_tiles = (numOfCols + TILE_SIZE - 1) / TILE_SIZE;

    /**************************** LOAD PHASE ***************************************************** */
    // Our for loop iterates over global deciding how many times we must slide across Q and K to capture the entire row
    for (int tileId = 0; tileId < num_tiles; tileId++) {

        // Our tile iterator * HowIsOurTile + WhereWeAreInTheTile, this explores column wise
        int q_col = tileId * TILE_SIZE + threadIdx.x;

        //Ensure our we never go outbounds vertically, Ensure we never go out of bounds horizontally
        if (row < seq_q && q_col < numOfCols) {
            //If within bounds, add find the global position of our data, add it to Q_tile shared memory
            Q_tile[threadIdx.y][threadIdx.x] = Q[row * numOfCols + q_col];
        } else {
            //If not, pad with 0.0f to ensure square calculations.
            Q_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }


        // We do the same for our K tile, this time we go row wise rather than column

        //We use our tileId iterator to slide down our K matrix
        int k_row = tileId * TILE_SIZE + threadIdx.x;

        //We use our blockIdx to ensure we stay fixed across a column
        int k_col = blockIdx.x * TILE_SIZE + threadIdx.y;

        //Bounds checks
        if (k_col < seq_k && k_row < numOfCols) {
            //We iterate through our current column, going down row by row, we are currently uncoalesced going down row
            K_tile[threadIdx.x][threadIdx.y] = K[k_col * numOfCols + k_row];
        } else {
            K_tile[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

    /**************************** COMPUTE PHASE ***************************************************** */
        //Calculate our loop at compile time, eliminates branching overhead and needing to track value k
        #pragma unroll
        
        //Iterate within our current tile and multiply Q_tile with K_tile
        for (int k = 0; k < TILE_SIZE; k++) {
            //We go across the row in Q, and down the columns in K_tile 
            //We're accessing Q_tile column wise, must be padded by +1
            softmax_numerator += Q_tile[threadIdx.y][k] * K_tile[k][threadIdx.x];
        }

        __syncthreads();
    }

    /**************************** STORE PHASE ***************************************************** */
    //Ensure our bounds are within
    if (row < seq_q && col < seq_k) {
        //Store our accumulated num in our output matrix S
        S[row * seq_k + col] = softmax_numerator;
    }
}







template <int BLOCK_SIZE>

__global__ void scale_and_softmax(
    float* __restrict__ S,
    int seq_q, int seq_k,
    float constant_scale
) {
/**************************** Initialize row pointers & Shared memory***************************************************** */
    //Each block within our grid processes one row of data
    int row = blockIdx.x;
    //Ensure we are not out of bounds
    if (row >= seq_q) return;

    // Points to the start of our current row
    //             WhichRow * SizeOfRow + OffsetWithinRow
    float* row_ptr = row * seq_k + S;

    __shared__ float smem_max[BLOCK_SIZE];
    __shared__ float smem_sum[BLOCK_SIZE];


/**************************** (LOAD/COMPUTE PHASE) Get the max number in our row ***************************************************** */
    //Ensure our first max number will always be larger than our initialized value
    float max_number = -INFINITY;

    //Ensure our threadIdx.x is within bounds of seq_k. Important for tail handling.
    if (threadIdx.x < seq_k) {
        //Multiply our row value by our constant
        row_ptr[threadIdx.x] *= constant_scale;
        //Our threads (0-1023 or BLOCK_SIZE) each put their respective S value into its register 
        max_number = row_ptr[threadIdx.x];
    }

    //We store our register value in shared memory
    smem_max[threadIdx.x] = max_number;
    __syncthreads();

    //We use tree reductions to find our maximum number within the row
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem_max[threadIdx.x] = fmaxf(smem_max[threadIdx.x], smem_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    //We store our absolute row max in our registers
    float row_max = smem_max[0];

/**************************** (LOAD/COMPUTE PHASE) Get our total sum across the row ***************************************************** */

    //Initialize our running total
    float runningTotal = 0.0f;
    
    //Bounds check
    if (threadIdx.x < seq_k) {
        //We do e^(curretNum - row_max) to ensure our number does not scale beyond what we can represent
        runningTotal = expf(row_ptr[threadIdx.x] - row_max);
        //Store our running in respective row_ptr location (based on threadIdx.x)
        row_ptr[threadIdx.x] = runningTotal;
    }

    //Each thread stores its running sum to its respective smem location
    smem_sum[threadIdx.x] = runningTotal;
    __syncthreads();

    //Tree reduction, we add up all of our running sums across the block
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem_sum[threadIdx.x] += smem_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    //Store are total running sum in our local row_sum
    float row_sum = smem_sum[0];

/**************************** (STORE PHASE) Calculate our softmax values ***************************************************** */

    //We compute each row_ptrs softmax value (0 - 1).
    if (threadIdx.x < seq_k) {
        row_ptr[threadIdx.x] /= row_sum;
    }
}









__global__ void attention_v3_online(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O,
    int seq_q, int seq_k, int d,
    float constant_scale
) {
/**************************** Initialize row pointers & Shared memory***************************************************** */
    //For K_tile, we implicitly transpose our K matrix by going down each column (rather than across the row), so we require padding to avoid bank conflicts
    __shared__ float Q_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float K_tile[TILE_SIZE][TILE_SIZE+1];
    __shared__ float V_tile[TILE_SIZE][TILE_SIZE];
    // S_tile stores the completed dot product scores in shared memory so every thread
    // in the same row can read every score during the S*V multiply
    __shared__ float S_tile[TILE_SIZE][TILE_SIZE];

    //Our respective current row/col using WhereAmI * howBigAmI + WhereAmIWithin
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    // col must include blockIdx.x offset so each x-block writes to the correct d-columns of O
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    //Our thread local accumulated value
    float softmax_numerator = 0.0f;
    //Ensure our maximum number is always larger than our initialized value
    float max_number = -INFINITY;
    //Our local runningTotal
    float runningTotal = 0.0f;

    //We determine how many times we must move our TILE to fully occupy our sequence size dimension
    int num_k_tiles = (seq_k + TILE_SIZE - 1) / TILE_SIZE;

    for (int tileId_LongSide = 0; tileId_LongSide < num_k_tiles; tileId_LongSide++) {

        //Initialize our S_tile with 0.0f to later be added onto
        S_tile[threadIdx.y][threadIdx.x] = 0.0f;

        //We determine how many times we must move our TILE to fully occupy our smaller D dimension. Note this is the dimension that gets reduced in our GEMM 
        int num_d_tiles = (d + TILE_SIZE - 1) / TILE_SIZE;

        for (int tileId_ShortSide = 0; tileId_ShortSide < num_d_tiles; tileId_ShortSide++) {

/**************************** LOAD our Q values into Shared Memory ***************************************************** */

            //Our iterator is used to traverse the columns of matrix Q
            //       (Our Current Tile) * (Size of Tile) + (Where we are in our tile)
            int q_col = tileId_ShortSide * TILE_SIZE + threadIdx.x;

            //Ensure we are within bounds
            if (row < seq_q && q_col < d) {
                //We load our Q value into our shared memory
                //                                (Which row) (How big is our row) + (Where are we within the row)
                Q_tile[threadIdx.y][threadIdx.x] = Q[row * d + q_col];
            } else {
                //Pad if value out of bounds for easier later calculation
                Q_tile[threadIdx.y][threadIdx.x] = 0.0f;
            }

/**************************** LOAD our K values into Shared Memory ***************************************************** */

            // Our iterator is used to traverse the columns of K
            //          (Which tile) * (How big is the tile) + (Where are we within the tile)
            int k_d_idx   = tileId_ShortSide * TILE_SIZE + threadIdx.x;

            // Which block are we in column wise, and where are we within the block row wise
            //            (Which block are we in column wise) * (How big our selection within this columnn is) + (Where we are within our tile)
            int k_seq_idx = tileId_LongSide * TILE_SIZE + threadIdx.y;

            //Ensure we are within bounds
            if (k_seq_idx < seq_k && k_d_idx < d) {
                //We implicitly transpose our matrix to later be accessed column wise
                //         (Which block are we in column wise within K) * (How big is said block) + (Which column are we in within the block)
                K_tile[threadIdx.x][threadIdx.y] = K[k_seq_idx * d + k_d_idx];
            } else {
                K_tile[threadIdx.x][threadIdx.y] = 0.0f;
            }

            __syncthreads();

/**************************** COMPUTE Dot product of Q and K values within Shared Memory ***************************************************** */

            //We iterate within our tile
            for (int k = 0; k < TILE_SIZE; k++) {
                //Each K iteration: In our S_tile, we store a row of DOT products of a single row of Q, and multiple columns of K
                S_tile[threadIdx.y][threadIdx.x] += Q_tile[threadIdx.y][k] * K_tile[k][threadIdx.x];

                //Once the loop is done, we have stored the DOT product of all Q_tile and K_tile in S_tile
            }

            __syncthreads();
        }

        // We scale our values once we're done. This allows saves us computation vs if we were to do it within the loop
        S_tile[threadIdx.y][threadIdx.x] *= constant_scale;

        // Mask out-of-bounds K positions so they contribute zero weight to softmax
        // Without this, zero-padded positions produce exp(0 - max) > 0, inflating runningTotal
        int k_col_global = tileId_LongSide * TILE_SIZE + threadIdx.x;
        if (k_col_global >= seq_k) {
            S_tile[threadIdx.y][threadIdx.x] = -INFINITY;
        }

        __syncthreads();

/**************************** COMPUTE our max number and running sum within our TILE ***************************************************** */

        //Initalize our max to our the lowest value possible to ensure our value will always replace our set tile_max
        //Used for our local tile max
        float tile_max = -INFINITY;

        //We iterate through our tile
        for (int k = 0; k < TILE_SIZE; k++) {
            //We compare our current tile_max with our current S_tile value.
            //                              [Our current row fixed][our k iterates through columns]
            tile_max = fmaxf(tile_max, S_tile[threadIdx.y][k]);
            //NOTE ^
            // If we were to use [threadIdx.y][threadIdx.x] we would just be looking at our own values since our threadIdx.x is always the same value from 
            // the perspective of the thread itself. Using K allows us to iterate over all values over the row in our tile_max
        }

        //Our max across ALL tiles thus far <--
        float new_max = fmaxf(max_number, tile_max);

        // We keep track our running total using e^(curretNum - maxNum) to ensure no numerical overflow
        // If we come across a new max, we use bridge A * B + C to ensure all our old max values are scaled correctly
        float correction = expf(max_number - new_max);

        //-------> When we come across a new max, we must rescale both our runningTotal AND our accumulated weighted values. The latter requires us it materialize our V matrix.

    //---------------------------------------------------- Our A * B for runningTotal
        //We bridge our runningTotal thus far
        runningTotal *= correction;
    //---------------------------------------------------

    //--------------------------------------------------- Our + C for runningTotal
        for (int k = 0; k < TILE_SIZE; k++) {
            runningTotal += expf(S_tile[threadIdx.y][k] - new_max);
        }
    //---------------------------------------------------
        //Our new max is stored locally
        max_number = new_max;

/**************************** LOAD our matrix V into shared memory ***************************************************** */
       
    //----------------------------------------------------- Our A * B for softmax_numerator
        softmax_numerator *= correction;
    //-----------------------------------------------------


    //----------------------------------------------------- Our + C for softmax_numerator
        
        // Our row selector within our tile for matrix V
        //       (Current Tile) * (Size of Tile) + (Which row are we in within that tile)
        int v_row = tileId_LongSide * TILE_SIZE + threadIdx.y;
        // Our col selector within our tile for matrix V
        //       (We use our threadIdx.x which is 0 - (BLOCK_SIZE - 1)) -> For an example if our block size is 1024, threadIdx.x would provide values 0 - 1023
        int v_col = col;

        //Bounds checks
        if (v_row < seq_k && v_col < d) {
            // We store our current value in the V matrix inn our shared memory (regular, no transpose)
            //                               (Which row are we in) * (Size of row) + (Which column are we in within the row)
            V_tile[threadIdx.y][threadIdx.x] = V[v_row * d + v_col];
        } else {
            //Pad value if out of bounds so we can calculate more easily without needing to worry about being out of bounds
            V_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

/**************************** COMPUTE (Q * K) * V ***************************************************** */

        //We iterate within our current tile
        for (int k = 0; k < TILE_SIZE; k++) {
            // We use e^(curretNum - maxNum) to avoid numerical overflow
            // S_tile[threadIdx.y][k] has our DOT product of how similar our respective a row of Q and a column of K are to each other
            // V_tile has what our Q * K DOT product means
            // We multiply S[k] and our V_Tile value together so that we can say our V_tile value matters this much to our Q given our similar Q and K are.
            //              Multiply our e^(currentNum - max) * (Iterate through our V_Tile, k stays fixed on a row, threadIdx.x goes across the row) -> No shared memory padding needed
            // Add our current value onto our accumulated weighted sum thus far
            softmax_numerator += expf(S_tile[threadIdx.y][k] - max_number) * V_tile[k][threadIdx.x];
        }

        __syncthreads();
    }
    //-----------------------------------------------------------------

/**************************** COMPUTE & STORE our Softmax value ***************************************************** */

    //Bounds check, are row is checked by our seq_q (long side) and our column is checked by our d size (short side)
    if (row < seq_q && col < d) {
        // We store our softmax value into the respective O matrix location
        // (Our current row within O) * (Size of Row) + (Which column within row) = softmax of accumulatedValue/runningTotal
        O[row * d + col] = softmax_numerator / runningTotal;
    }
}


/**
 * This is very interesting! I see why they call it the KV cache and why we only maintain just the key and values.
 * 
 * I assume this is because once we do our Q * K DOT product, the value we store tells us how similar our Q and K are to each other
 * this means we no longer need Q because we already know our similar our key is to our query.
 * 
 * If our key somehow changes dynamically for the same Q, then we would need to reupdate it so we would once again require Q, but as far as I know that is not the case!
 * 
 */















/* Multi-Headed Attention
    We incorporate a new dimension Z within our block. I am new to this so this will be interesting none the less.
    It seems each head does not require communication among one another.......

    Tensor Core is essentially a unrolling the loop on the third dimension. Earlier in my parking lot I mentioned that the 
    3rd dimension might not just be a spatial dimension like the first two, but rather a dimension of time. I say this because when I look at 3D convolutions,
    The first two for loops give us the pixels x and y, but the third for loop gives us seperate frames, so a time dimension if you will.

    I made a LinkedIn post about this and why I think this at the moment https://www.linkedin.com/feed/update/urn:li:activity:7440774158847913984/?originTrackingId=bfJ0MRzY8tLPNzV%2FDZiFgg%3D%3D

    The thing is, if this is true, that means that this use of our 3rd dimension is essentially us just doing the same attention we did in the kernel before, but just the size of our Z dimension at once.

    Again, if this is true, that means Tensor Cores were quite literally made for this, or rather, multi headed attention was quite literally made for Tensor Cores!!
 

    Does this mean that if our Z dimension is essentially a hardware level unroll, does that mean for any execution heirarchy if we re-organize its dimensional structure
    to reach this 3rd dimension, can any kernel take advantage of these Tensor Cores? One can say to reach this 3rd dimension its computationally expensive where our registers
    must hold these values, or even the indexes required to reach said dimension hence the question becomes

    Does the increased register use justify needing Tensor Cores, or can the kernel do just fine with the increased occupancy/warp level paralelism that comes with low register counts.

    I suppose for memory bound kernels I believe that we want more occupancy for latency hiding, so for a memory bound kernel if we intentionally increase register pressure to reach this
    3rd dimemnsion, its rather not worth it

    For compute bound kernels, we may be able to trade register pressure for us to process our data from a 3rd dimension (Tensor Cores) since we trade higher occupancy/latency hiding for pure
    compute power.

    ------------------------------------------------

    The externel PyTorch test taught me that I had assumed both our passed Q K V matrices we're all square. 

    This shows a gap in my kernel process where I should first write down my assumptions, so that they can be tackled through edge case handling.

    Knowing this, I cannot assume that Q K and V will be proper square matrices, and my code should handle such cases

    seq_q = Number of query tokens
    seq_k = Number of keys
    d_head = size of each token's vector within one attention head


    DIMENSIONS OF OUR INPUT
    Q -> (seq_q,d_head)
    K -> (seq_k, d_head)
    V -> (seq_k, d_head)
    O -> (seq_q, d_head)


    - Load our Q tile
    - Load our K tile
    - DOT product of Q * K (transposed), (seq_q * d_head) * (seq_k * d_head), our d_head gets reduced, meaning our DOT tells us how similar Q's and K's d_head was to each other, stored in S_tile
    - We use our DOT product values stored in S_tile to determine our maxNum and runningSum
    - We load our matrix V for our + C term to scale runningSum
    - Compute and store weighted normalized softmax value

*/

__global__ void attention_v4_multihead(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O,
    int batch_size, int num_heads,
    int seq_q, int seq_k, int d_head,
    float constant_scale
) {
/**************************** Initialize row pointers & Shared memory***************************************************** */
    __shared__ float Q_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float K_tile[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float V_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float S_tile[TILE_SIZE][TILE_SIZE];

    //This is similar to shared memory where our Flatten -> Stride -> Unflatten required us to use 
    // the division to find our row, and modulus to find our column.
    // Is this Z dimension similar to shared memory thats pooled?
    // Because on the X and Y dimension, we use a simple A * B + C affline map, so why does this require us to use / and % to get our row/col?

    //We use our 1D given address of blockIdx.z to derive a 2D coordinate. The use of / and % allows us to unflatten from 1D to 2D.
    // This is a generalized pattern!!

    // A * B + C to flatten a 2D index into a 1D address
    // Using / and % expands a 1D address to a 2D index!
    
    // A * B + C means we go from the logical world to the hardware world!
    // / and % means we go from the hardware world to our logical world!
    // These operations are inverses of each other!

    int batch_idx = blockIdx.z / num_heads; // Our row within the Z dimension
    int head_idx  = blockIdx.z % num_heads; // Our column within the Z dimension

    // (Size of sequence_q) * (Size of d_head) -> Size of each column in our 3rd dimension
    int head_stride_q_o = seq_q * d_head; // Our Q and O match each other's dimensions
    int head_stride_k_v = seq_k * d_head; // Our K and V match each other's dimensions

    // (Number of Heads) * (How big each Head is) -> Size of each row (made up of our columns) in our 3rd dimension
    int batch_stride_q_o = num_heads * head_stride_q_o;
    int batch_stride_k_v = num_heads * head_stride_k_v;

    //           Execution     * Size in Memory +      Execution       * Size in Memory + Execution * 1 = Uses our Coordinate * Stride recursive formula from our Two Tree Framework
    //    Current Batch Column * Size of Batch  + Current Head Column  *  Size of Head  +    Offset Q   = Our global current address in our Q matrix
    const float* Q_head = batch_idx * batch_stride_q_o + head_idx * head_stride_q_o + Q;
    const float* K_head = batch_idx * batch_stride_k_v + head_idx * head_stride_k_v + K; // = Our global current address in our K matrix
    const float* V_head = batch_idx * batch_stride_k_v + head_idx * head_stride_k_v + V; // = Our global current address in our V matrix
    float*       O_head = batch_idx * batch_stride_q_o + head_idx * head_stride_q_o + O; // = Our global current address in our O matrix


    //     Our current row block * Size of our tile + Which row we are in the tile
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    //     Our current column block *  Size of our tile + Which column are we in
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;


    float softmax_numerator = 0.0f;
    float max_number  = -INFINITY;
    float runningTotal = 0.0f;

    // How many times does our tile need to slide to across the long side of our matrix 
    int num_k_tiles = (seq_k + TILE_SIZE - 1) / TILE_SIZE;

    // We iterate our tile over the long side
    for (int tileId_LongSide = 0; tileId_LongSide < num_k_tiles; tileId_LongSide++) {

        //Initialize our S_tile with 0.0f to ensure later accumulate
        S_tile[threadIdx.y][threadIdx.x] = 0.0f;

        // How many times does our tile need to slide across the short side of our matrix
        int num_d_tiles = (d_head + TILE_SIZE - 1) / TILE_SIZE;

        // We iterate our tile over the short side
        for (int tileId_ShortSide = 0; tileId_ShortSide < num_d_tiles; tileId_ShortSide++) {

/**************************** LOAD our Q values into Shared Memory ***************************************************** */
            // Our current column with the Q matrix, which we are iterating over
            int q_col = tileId_ShortSide * TILE_SIZE + threadIdx.x;
            
            //Bounds check
            if (row < seq_q && q_col < d_head)
                //Input our Q matrix value into our Q_tile
                //                                     Which Row * How big is the row + Which column within the row
                Q_tile[threadIdx.y][threadIdx.x] = Q_head[row * d_head + q_col];
            else
                //Pad value
                Q_tile[threadIdx.y][threadIdx.x] = 0.0f;

/**************************** LOAD our K values into Shared Memory ***************************************************** */
            //Which column are we in the K matrix, iterator
            int k_d_idx   = tileId_ShortSide * TILE_SIZE + threadIdx.x;
            //Which row are we in the K matrix, iterator
            int k_seq_idx = tileId_LongSide  * TILE_SIZE + threadIdx.y;

            //Bounds check
            if (k_seq_idx < seq_k && k_d_idx < d_head)
                //Implicit transpose storing in shared memory, (Which row are we in) * (How big is each row) + (Which column)
                K_tile[threadIdx.x][threadIdx.y] = K_head[k_seq_idx * d_head + k_d_idx];
            else
                //Pad values
                K_tile[threadIdx.x][threadIdx.y] = 0.0f;

            __syncthreads();

/**************************** COMPUTE Q * K TILE DOT Product ***************************************************** */       
            //Iterate over our tile
            for (int k = 0; k < TILE_SIZE; k++)
                //Store our DOT product in S_tile
                //                             Q_tile: Our row is fixed, k iterates across columns
                //                             K_tile: Our row iterates, threadIdx.x iterates at the warp level. Note, while our row iterates, it says fixed while our threadIdx.x iterates across the column (parallelized)
                S_tile[threadIdx.y][threadIdx.x] += Q_tile[threadIdx.y][k] * K_tile[k][threadIdx.x];

            __syncthreads();
        }

        //Apply our constant scale to ensure no numerical overflow
        S_tile[threadIdx.y][threadIdx.x] *= constant_scale;


        // Iterate across our K matrix's columns
        int k_col_global = tileId_LongSide * TILE_SIZE + threadIdx.x;
        // Any global column outside of our seq_k is set to -infinity as not to influence our softmax score, ensuring proper calculation for non square dimensions
        if (k_col_global >= seq_k)
            S_tile[threadIdx.y][threadIdx.x] = -INFINITY;

        __syncthreads();

/**************************** COMPUTE our max number and running sum within our TILE ***************************************************** */
        // Ensure our number will always override our intialized value
        float tile_max = -INFINITY;

        //Iterate over our tile
        for (int k = 0; k < TILE_SIZE; k++)
            //If currentNum is larger than our current, replace our tile_max
            tile_max = fmaxf(tile_max, S_tile[threadIdx.y][k]);

        //Our max across all tiles thus far
        float new_max   = fmaxf(max_number, tile_max);

        // Our A * B + C correction, specifically the A * B bridge. We must update both runningTotal and softmax_numerator. The latter requires us to materialize tile V
        float correction = expf(max_number - new_max);

    //---------------------------------------------------- Our A * B for runningTotal
        // We apply our A * B bridge from our old max to our new max, now we must add our current value C
        runningTotal *= correction;

        //We iterate over our tile
        for (int k = 0; k < TILE_SIZE; k++)
            //We add our current values to our running total, this is our + C
            runningTotal += expf(S_tile[threadIdx.y][k] - new_max);


        //Update our local max to our new max, will be used to compare against in future iterations
        max_number = new_max;

/**************************** LOAD our matrix V into shared memory ***************************************************** */        
    //---------------------------------------------------- Our A * B for softmax_numerator
        // We apply our A * B bridge from our old max to our new max, now we must add our current value C
        // Note, this is technically not included in our LOAD of V tile, and could have been done at the same time as our runningTotal *= correction,
        // but for clarity sakes, we can bring it down here as it groups our A * B + C operation for softmax_numerator
        softmax_numerator *= correction;
    //------------------------------------------------------
        // We iterate over our v_row
        int v_row = tileId_LongSide * TILE_SIZE + threadIdx.y;
        // Our current column in our V tile
        int v_col = col;
        
        //Bounds check
        if (v_row < seq_k && v_col < d_head)
            //We store our V_tile in shared memory,   (Current Row) * (Size of Row) + (Column in our row)
            V_tile[threadIdx.y][threadIdx.x] = V_head[v_row * d_head + v_col];
        else
            //Pad values
            V_tile[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        //We iterate over our tile size
        for (int k = 0; k < TILE_SIZE; k++)
            //We add our current values to our softmax_numerator, we are doing 
            //                                our e^(S_tile unweighted DOT product - max)
            //                                our V_tile[iterate down row][parallelize across row, column wise] 
            // This gives us our updated + C to add to softmax_numerator  
            softmax_numerator += expf(S_tile[threadIdx.y][k] - max_number) * V_tile[k][threadIdx.x];

        __syncthreads();
    }

/**************************** COMPUTE & STORE our Softmax value ***************************************************** */
    //Bounds check
    if (row < seq_q && col < d_head)
        //We output our 0-1 softmax value to our O matrix
        O_head[row * d_head + col] = softmax_numerator / runningTotal;
}





















/* CODE BELOW IS AI GENERATED */


/**************************** Helpers ***************************************************** */

// Helpers and CPU references
static void fill_rand(float* buf, int n) {
    for (int i = 0; i < n; i++) buf[i] = ((float)rand()/RAND_MAX) - 0.5f;
}

static float max_abs_error(const float* a, const float* b, int n) {
    float e = 0.f;
    for (int i = 0; i < n; i++) e = fmaxf(e, fabsf(a[i]-b[i]));
    return e;
}

// CPU reference for multi‑head attention (used by run_test)
static void multihead_attention_cpu(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int num_heads, int seq_q, int seq_k, int d_head, float scale
) {
    int hs_q=seq_q*d_head, bs_q=num_heads*hs_q;
    int hs_kv=seq_k*d_head, bs_kv=num_heads*hs_kv;
    float* S = (float*)malloc(seq_q*seq_k*sizeof(float));
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            const float* Qh = Q + b*bs_q  + h*hs_q;
            const float* Kh = K + b*bs_kv + h*hs_kv;
            const float* Vh = V + b*bs_kv + h*hs_kv;
            float*       Oh = O + b*bs_q  + h*hs_q;
            for (int i = 0; i < seq_q; i++)
                for (int j = 0; j < seq_k; j++) {
                    float s = 0.f;
                    for (int k = 0; k < d_head; k++) s += Qh[i*d_head+k]*Kh[j*d_head+k];
                    S[i*seq_k+j] = s*scale;
                }
            for (int i = 0; i < seq_q; i++) {
                float mx=-INFINITY, sum=0.f;
                for (int j = 0; j < seq_k; j++) if (S[i*seq_k+j]>mx) mx=S[i*seq_k+j];
                for (int j = 0; j < seq_k; j++) { S[i*seq_k+j]=expf(S[i*seq_k+j]-mx); sum+=S[i*seq_k+j]; }
                for (int j = 0; j < seq_k; j++) S[i*seq_k+j]/=sum;
            }
            for (int i = 0; i < seq_q; i++)
                for (int d = 0; d < d_head; d++) {
                    float s = 0.f;
                    for (int j = 0; j < seq_k; j++) s += S[i*seq_k+j]*Vh[j*d_head+d];
                    Oh[i*d_head+d] = s;
                }
        }
    }
    free(S);
}

// ----------------------------------------------------------------------
// Test functions for each kernel
static void test_qkt_tiled() {
    const int seq_q = 64, seq_k = 64, d = 32;
    int total_q = seq_q * d;
    int total_k = seq_k * d;
    float *h_Q = (float*)malloc(total_q * sizeof(float));
    float *h_K = (float*)malloc(total_k * sizeof(float));
    float *h_S = (float*)malloc(seq_q * seq_k * sizeof(float));
    float *h_S_ref = (float*)malloc(seq_q * seq_k * sizeof(float));

    fill_rand(h_Q, total_q);
    fill_rand(h_K, total_k);

    float *d_Q, *d_K, *d_S;
    cudaMalloc(&d_Q, total_q * sizeof(float));
    cudaMalloc(&d_K, total_k * sizeof(float));
    cudaMalloc(&d_S, seq_q * seq_k * sizeof(float));
    cudaMemcpy(d_Q, h_Q, total_q * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, total_k * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((seq_k + TILE_SIZE - 1) / TILE_SIZE,
                (seq_q + TILE_SIZE - 1) / TILE_SIZE);
    qkt_tiled<<<blocks, threads>>>(d_Q, d_K, d_S, seq_q, seq_k, d);
    cudaDeviceSynchronize();
    cudaMemcpy(h_S, d_S, seq_q * seq_k * sizeof(float), cudaMemcpyDeviceToHost);

    // CPU reference: Q * K^T
    for (int i = 0; i < seq_q; i++) {
        for (int j = 0; j < seq_k; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d; k++)
                sum += h_Q[i * d + k] * h_K[j * d + k];
            h_S_ref[i * seq_k + j] = sum;
        }
    }

    float err = max_abs_error(h_S, h_S_ref, seq_q * seq_k);
    printf("qkt_tiled: max_err = %.2e %s\n", err, err < 1e-5 ? "PASS" : "FAIL");

    free(h_Q); free(h_K); free(h_S); free(h_S_ref);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_S);
}

static void test_scale_and_softmax() {
    const int seq_q = 64, seq_k = 64, d = 32;
    const float scale = 1.0f / sqrtf((float)d);
    int total_q = seq_q * d;
    int total_k = seq_k * d;
    float *h_Q = (float*)malloc(total_q * sizeof(float));
    float *h_K = (float*)malloc(total_k * sizeof(float));
    fill_rand(h_Q, total_q);
    fill_rand(h_K, total_k);

    float *h_S = (float*)malloc(seq_q * seq_k * sizeof(float));
    float *h_S_ref = (float*)malloc(seq_q * seq_k * sizeof(float));

    // Compute S = Q * K^T on CPU
    for (int i = 0; i < seq_q; i++) {
        for (int j = 0; j < seq_k; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d; k++)
                sum += h_Q[i * d + k] * h_K[j * d + k];
            h_S_ref[i * seq_k + j] = sum * scale;
        }
    }

    // Copy to device
    float *d_S;
    cudaMalloc(&d_S, seq_q * seq_k * sizeof(float));
    cudaMemcpy(d_S, h_S_ref, seq_q * seq_k * sizeof(float), cudaMemcpyHostToDevice);

    const int BLOCK_SIZE = 256; // Must be >= seq_k (64)
    dim3 threads(BLOCK_SIZE);
    dim3 blocks(seq_q);
    scale_and_softmax<BLOCK_SIZE><<<blocks, threads>>>(d_S, seq_q, seq_k, scale);
    cudaDeviceSynchronize();
    cudaMemcpy(h_S, d_S, seq_q * seq_k * sizeof(float), cudaMemcpyDeviceToHost);

    // CPU reference: row-wise softmax
    for (int i = 0; i < seq_q; i++) {
        float max_val = -INFINITY;
        for (int j = 0; j < seq_k; j++)
            max_val = fmaxf(max_val, h_S_ref[i * seq_k + j]);
        float sum = 0.0f;
        for (int j = 0; j < seq_k; j++) {
            h_S_ref[i * seq_k + j] = expf(h_S_ref[i * seq_k + j] - max_val);
            sum += h_S_ref[i * seq_k + j];
        }
        for (int j = 0; j < seq_k; j++)
            h_S_ref[i * seq_k + j] /= sum;
    }

    float err = max_abs_error(h_S, h_S_ref, seq_q * seq_k);
    printf("scale_and_softmax: max_err = %.2e %s\n", err, err < 1e-5 ? "PASS" : "FAIL");

    free(h_Q); free(h_K); free(h_S); free(h_S_ref);
    cudaFree(d_S);
}

static void test_attention_v3() {
    const int seq_q = 64, seq_k = 64, d = 32;
    const float scale = 1.0f / sqrtf((float)d);
    int total_q = seq_q * d;
    int total_kv = seq_k * d;

    float *h_Q = (float*)malloc(total_q * sizeof(float));
    float *h_K = (float*)malloc(total_kv * sizeof(float));
    float *h_V = (float*)malloc(total_kv * sizeof(float));
    float *h_O = (float*)malloc(total_q * sizeof(float));
    float *h_ref = (float*)malloc(total_q * sizeof(float));

    fill_rand(h_Q, total_q);
    fill_rand(h_K, total_kv);
    fill_rand(h_V, total_kv);

    // CPU reference: full attention (naive)
    float *S = (float*)malloc(seq_q * seq_k * sizeof(float));
    for (int i = 0; i < seq_q; i++) {
        for (int j = 0; j < seq_k; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d; k++)
                sum += h_Q[i * d + k] * h_K[j * d + k];
            S[i * seq_k + j] = sum * scale;
        }
    }
    for (int i = 0; i < seq_q; i++) {
        float max_val = -INFINITY;
        for (int j = 0; j < seq_k; j++) max_val = fmaxf(max_val, S[i * seq_k + j]);
        float sum = 0.0f;
        for (int j = 0; j < seq_k; j++) {
            S[i * seq_k + j] = expf(S[i * seq_k + j] - max_val);
            sum += S[i * seq_k + j];
        }
        for (int j = 0; j < seq_k; j++) S[i * seq_k + j] /= sum;
    }
    for (int i = 0; i < seq_q; i++) {
        for (int k = 0; k < d; k++) {
            float val = 0.0f;
            for (int j = 0; j < seq_k; j++)
                val += S[i * seq_k + j] * h_V[j * d + k];
            h_ref[i * d + k] = val;
        }
    }
    free(S);

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, total_q * sizeof(float));
    cudaMalloc(&d_K, total_kv * sizeof(float));
    cudaMalloc(&d_V, total_kv * sizeof(float));
    cudaMalloc(&d_O, total_q * sizeof(float));
    cudaMemcpy(d_Q, h_Q, total_q * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, total_kv * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, total_kv * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((d + TILE_SIZE - 1) / TILE_SIZE,
                (seq_q + TILE_SIZE - 1) / TILE_SIZE);
    attention_v3_online<<<blocks, threads>>>(d_Q, d_K, d_V, d_O, seq_q, seq_k, d, scale);
    cudaDeviceSynchronize();
    cudaMemcpy(h_O, d_O, total_q * sizeof(float), cudaMemcpyDeviceToHost);

    float err = max_abs_error(h_O, h_ref, total_q);
    printf("attention_v3_online: max_err = %.2e %s\n", err, err < 1e-5 ? "PASS" : "FAIL");

    free(h_Q); free(h_K); free(h_V); free(h_O); free(h_ref);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
}

// The existing run_test function for multi‑head kernel
// (PASS < 1e-3 | FAIL (numerical inaccuracy) 1e-3..1e-2 | FAIL (square/non-square) > 1e-2)
static void run_test(const char* label, int batch_size, int num_heads,
                     int seq_q, int seq_k, int d_head, int square) {
    float scale   = 1.0f / sqrtf((float)d_head);
    int total_q   = batch_size * num_heads * seq_q * d_head;
    int total_kv  = batch_size * num_heads * seq_k * d_head;

    float *h_Q=(float*)malloc(total_q *sizeof(float)), *h_K=(float*)malloc(total_kv*sizeof(float));
    float *h_V=(float*)malloc(total_kv*sizeof(float)), *h_O=(float*)malloc(total_q *sizeof(float));
    float *h_ref=(float*)malloc(total_q*sizeof(float));
    fill_rand(h_Q,total_q); fill_rand(h_K,total_kv); fill_rand(h_V,total_kv);

    float *d_Q,*d_K,*d_V,*d_O;
    cudaMalloc(&d_Q,total_q *sizeof(float)); cudaMalloc(&d_K,total_kv*sizeof(float));
    cudaMalloc(&d_V,total_kv*sizeof(float)); cudaMalloc(&d_O,total_q *sizeof(float));
    cudaMemcpy(d_Q,h_Q,total_q *sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_K,h_K,total_kv*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_V,h_V,total_kv*sizeof(float),cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE,TILE_SIZE);
    dim3 blocks((d_head+TILE_SIZE-1)/TILE_SIZE,(seq_q+TILE_SIZE-1)/TILE_SIZE,batch_size*num_heads);

    printf("\n=== %s | batch=%d heads=%d seq_q=%d seq_k=%d d_head=%d ===\n",
           label, batch_size, num_heads, seq_q, seq_k, d_head);
    printf("Grid: (%d,%d,%d)  Block: (%d,%d)\n",
           blocks.x,blocks.y,blocks.z,threads.x,threads.y);

    attention_v4_multihead<<<blocks,threads>>>(
        d_Q,d_K,d_V,d_O, batch_size,num_heads,seq_q,seq_k,d_head,scale);
    cudaDeviceSynchronize();
    cudaMemcpy(h_O,d_O,total_q*sizeof(float),cudaMemcpyDeviceToHost);

    multihead_attention_cpu(h_Q,h_K,h_V,h_ref,batch_size,num_heads,seq_q,seq_k,d_head,scale);
    float e = max_abs_error(h_O,h_ref,total_q);

    const char* verdict;
    if      (e < 1e-3f) verdict = "PASS";
    else if (e < 1e-2f) verdict = "FAIL (numerical inaccuracy)";
    else                verdict = square ? "FAIL (square test)" : "FAIL (non-square test)";
    printf("  >> %-30s  max_err=%.2e\n", verdict, e);
    printf("  %s\n", (e < 1e-3f) ? "------------------------------------------------------------"
                                   : "============================================================");

    free(h_Q);free(h_K);free(h_V);free(h_O);free(h_ref);
    cudaFree(d_Q);cudaFree(d_K);cudaFree(d_V);cudaFree(d_O);
}

// ----------------------------------------------------------------------
// Main: runs all four test suites
int main() {
    srand(42);
    printf("\n========== Testing qkt_tiled ==========\n");
    test_qkt_tiled();
    printf("\n========== Testing scale_and_softmax ==========\n");
    test_scale_and_softmax();
    printf("\n========== Testing attention_v3_online ==========\n");
    test_attention_v3();

    printf("\n========== Testing attention_v4_multihead ==========\n");
    printf("\n==================== SQUARE TESTS (seq_q == seq_k) ====================\n");
    run_test("Square 1", 1,4,  64, 64, 32, 1);
    run_test("Square 2", 2,4,  48, 48, 32, 1);
    run_test("Square 3", 3,8,  96, 96, 64, 1);

    printf("\n==================== NON-SQUARE TESTS (seq_q != seq_k) ====================\n");
    run_test("Non-square 1 (seq_k > seq_q)", 1,4,  64,128, 32, 0);
    run_test("Non-square 2 (seq_k > seq_q)", 2,4,  48, 80, 32, 0);
    run_test("Non-square 3 (seq_k > seq_q)", 3,8,  96,160, 64, 0);
    run_test("Non-square 4 (seq_q > seq_k)", 1,4, 128, 64, 32, 0);
    run_test("Non-square 5 (seq_q > seq_k)", 2,4,  80, 48, 32, 0);
    run_test("Non-square 6 (seq_q > seq_k)", 3,8, 160, 96, 64, 0);
    run_test("Non-square 7 (d_head unaligned)", 2,4, 48, 80, 48, 0);

    return 0;
}

// ----------------------------------------------------------------------
// PyTorch extension wrapper (only when building with WITH_TORCH)
#ifdef WITH_TORCH
#include <torch/extension.h>

torch::Tensor attention_forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, float scale)
{
    int batch_size=Q.size(0), num_heads=Q.size(1), seq_q=Q.size(2), d_head=Q.size(3);
    int seq_k=K.size(2);
    torch::Tensor O = torch::empty_like(Q);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((d_head+TILE_SIZE-1)/TILE_SIZE,
                (seq_q +TILE_SIZE-1)/TILE_SIZE,
                batch_size * num_heads);

    attention_v4_multihead<<<blocks, threads>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(),
        V.data_ptr<float>(), O.data_ptr<float>(),
        batch_size, num_heads, seq_q, seq_k, d_head, scale
    );
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &attention_forward, "attention_v4_multihead CUDA");
}
#endif