
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
    int head_stride = seq_q * d_head;

    // (Number of Heads) * (How big each Head is) -> Size of each row (made up of our columns) in our 3rd dimension
    int batch_stride = num_heads * head_stride;

    //           Execution     * Size in Memory +      Execution       * Size in Memory + Execution * 1 = Uses our Coordinate * Stride recursive formula from our Two Tree Framework

    //    Current Batch Column * Size of Batch  + Current Head Column  *  Size of Head  +    Offset Q   = Our global current address in our Q matrix
    const float* Q_head = batch_idx * batch_stride + head_idx * head_stride + Q;
    const float* K_head = batch_idx * batch_stride + head_idx * head_stride + K; // = Our global current address in our K matrix
    const float* V_head = batch_idx * batch_stride + head_idx * head_stride + V; // = Our global current address in our V matrix
    float*       O_head = batch_idx * batch_stride + head_idx * head_stride + O; // = Our global current address in our O matrix


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
        softmax_numerator *= correction;

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

        for (int k = 0; k < TILE_SIZE; k++)
            softmax_numerator += expf(S_tile[threadIdx.y][k] - max_number) * V_tile[k][threadIdx.x];

        __syncthreads();
    }

    if (row < seq_q && col < d_head)
        O_head[row * d_head + col] = softmax_numerator / runningTotal;
}














/* CODE BELOW IS AI GENERATED */

// ==========================================================================
// CPU reference: full multi-head attention
// Layout: (batch, num_heads, seq, d_head) — same as the GPU kernel
// For each (batch, head) pair independently:
//   1. Q * K^T scaled
//   2. Row-wise softmax
//   3. Softmax weights * V
// ==========================================================================
void multihead_attention_cpu(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int num_heads, int seq_q, int seq_k, int d_head,
    float constant_scale
) {
    int head_stride  = seq_q * d_head;
    int batch_stride = num_heads * head_stride;

    // Temporary buffer for the (seq_q, seq_k) score matrix — one head at a time
    float* S = (float*)malloc(seq_q * seq_k * sizeof(float));

    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {

            // Navigate to this (batch, head) pair — same TTF formula as the GPU kernel
            const float* Q_head = Q + b * batch_stride + h * head_stride;
            const float* K_head = K + b * batch_stride + h * head_stride;
            const float* V_head = V + b * batch_stride + h * head_stride;
            float*       O_head = O + b * batch_stride + h * head_stride;

            // Q * K^T scaled: S[i][j] = dot(Q[i], K[j]) * scale
            for (int i = 0; i < seq_q; i++) {
                for (int j = 0; j < seq_k; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < d_head; k++)
                        sum += Q_head[i * d_head + k] * K_head[j * d_head + k];
                    S[i * seq_k + j] = sum * constant_scale;
                }
            }

            // Row-wise numerically stable softmax
            for (int i = 0; i < seq_q; i++) {
                float max_val = -INFINITY;
                for (int j = 0; j < seq_k; j++)
                    if (S[i * seq_k + j] > max_val) max_val = S[i * seq_k + j];
                float row_sum = 0.0f;
                for (int j = 0; j < seq_k; j++) {
                    S[i * seq_k + j] = expf(S[i * seq_k + j] - max_val);
                    row_sum += S[i * seq_k + j];
                }
                for (int j = 0; j < seq_k; j++)
                    S[i * seq_k + j] /= row_sum;
            }

            // Softmax weights * V: O[i][d] = sum_j S[i][j] * V[j][d]
            for (int i = 0; i < seq_q; i++) {
                for (int d = 0; d < d_head; d++) {
                    float sum = 0.0f;
                    for (int j = 0; j < seq_k; j++)
                        sum += S[i * seq_k + j] * V_head[j * d_head + d];
                    O_head[i * d_head + d] = sum;
                }
            }
        }
    }

    free(S);
}


// ==========================================================================
// Helper: fill a buffer with random floats in [-0.5, 0.5]
// ==========================================================================
void fill_rand(float* buf, int n) {
    for (int i = 0; i < n; i++)
        buf[i] = ((float)rand() / RAND_MAX) - 0.5f;
}


// ==========================================================================
// Helper: compute max absolute error between two float buffers
// ==========================================================================
float max_abs_error(const float* a, const float* b, int n) {
    float err = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(a[i] - b[i]);
        if (e > err) err = e;
    }
    return err;
}


// ==========================================================================
// Main: tests attention_v4_multihead across two configurations
//
// Test 1 — square, power-of-two dimensions, single batch
//   The "clean" case where no boundary guards are exercised.
//   Good for confirming the core logic is correct.
//
// Test 2 — non-square, non-power-of-two, multiple batches
//   seq_q != seq_k, neither a multiple of TILE_SIZE, batch_size > 1.
//   This exercises the out-of-bounds masking and the Z-dimension
//   delinearization across multiple batch elements simultaneously.
// ==========================================================================
int main() {
    srand(42);

    // ------------------------------------------------------------------
    // Test 1: single batch, square dimensions
    // ------------------------------------------------------------------
    {
        int batch_size = 1;
        int num_heads  = 4;
        int seq_q      = 64;
        int seq_k      = 64;
        int d_head     = 32;   // full d = num_heads * d_head = 128
        float scale    = 1.0f / sqrtf((float)d_head);

        // Total elements: batch * heads * seq * d_head
        int total_qkv = batch_size * num_heads * seq_q * d_head;
        int total_o   = batch_size * num_heads * seq_q * d_head;

        float* h_Q   = (float*)malloc(total_qkv * sizeof(float));
        float* h_K   = (float*)malloc(total_qkv * sizeof(float));
        float* h_V   = (float*)malloc(total_qkv * sizeof(float));
        float* h_O   = (float*)malloc(total_o   * sizeof(float));
        float* h_ref = (float*)malloc(total_o   * sizeof(float));

        fill_rand(h_Q, total_qkv);
        fill_rand(h_K, total_qkv);
        fill_rand(h_V, total_qkv);

        float *d_Q, *d_K, *d_V, *d_O;
        cudaMalloc(&d_Q, total_qkv * sizeof(float));
        cudaMalloc(&d_K, total_qkv * sizeof(float));
        cudaMalloc(&d_V, total_qkv * sizeof(float));
        cudaMalloc(&d_O, total_o   * sizeof(float));
        cudaMemcpy(d_Q, h_Q, total_qkv * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, h_K, total_qkv * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_V, total_qkv * sizeof(float), cudaMemcpyHostToDevice);

        // Grid: x tiles d_head, y tiles seq_q, z covers every (batch, head) pair
        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 blocks(
            (d_head + TILE_SIZE - 1) / TILE_SIZE,
            (seq_q  + TILE_SIZE - 1) / TILE_SIZE,
            batch_size * num_heads
        );

        printf("=== Test 1: batch=%d heads=%d seq_q=%d seq_k=%d d_head=%d ===\n",
               batch_size, num_heads, seq_q, seq_k, d_head);
        printf("Grid: (%d, %d, %d)  Block: (%d, %d)\n",
               blocks.x, blocks.y, blocks.z, threads.x, threads.y);

        attention_v4_multihead<<<blocks, threads>>>(
            d_Q, d_K, d_V, d_O,
            batch_size, num_heads, seq_q, seq_k, d_head, scale
        );

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return 1;
        }

        cudaMemcpy(h_O, d_O, total_o * sizeof(float), cudaMemcpyDeviceToHost);
        multihead_attention_cpu(h_Q, h_K, h_V, h_ref,
                                batch_size, num_heads, seq_q, seq_k, d_head, scale);

        float err_val = max_abs_error(h_O, h_ref, total_o);
        printf("Max absolute error vs CPU reference: %e\n", err_val);
        printf("Result: %s\n\n", err_val < 1e-3f ? "PASS ✓" : "FAIL ✗");

        free(h_Q); free(h_K); free(h_V); free(h_O); free(h_ref);
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    }

    // ------------------------------------------------------------------
    // Test 2: multiple batches, non-square non-power-of-two dimensions
    // This is the real stress test — it exercises:
    //   - Z delinearization across 2 * 4 = 8 independent computations
    //   - Out-of-bounds masking (seq_q=48 and seq_k=80 both non-multiples of 32)
    //   - seq_q != seq_k path through the kernel
    // ------------------------------------------------------------------
    {
        int batch_size = 2;
        int num_heads  = 4;
        int seq_q      = 48;   // not a multiple of TILE_SIZE (32)
        int seq_k      = 80;   // not a multiple of TILE_SIZE, and seq_q != seq_k
        int d_head     = 32;
        float scale    = 1.0f / sqrtf((float)d_head);

        // Note: K and V are (batch, heads, seq_k, d_head) while Q and O are (batch, heads, seq_q, d_head)
        // For simplicity we allocate with seq_q for Q/O and seq_k for K/V
        int total_q  = batch_size * num_heads * seq_q * d_head;
        int total_kv = batch_size * num_heads * seq_k * d_head;

        float* h_Q   = (float*)malloc(total_q  * sizeof(float));
        float* h_K   = (float*)malloc(total_kv * sizeof(float));
        float* h_V   = (float*)malloc(total_kv * sizeof(float));
        float* h_O   = (float*)malloc(total_q  * sizeof(float));
        float* h_ref = (float*)malloc(total_q  * sizeof(float));

        fill_rand(h_Q, total_q);
        fill_rand(h_K, total_kv);
        fill_rand(h_V, total_kv);

        float *d_Q, *d_K, *d_V, *d_O;
        cudaMalloc(&d_Q, total_q  * sizeof(float));
        cudaMalloc(&d_K, total_kv * sizeof(float));
        cudaMalloc(&d_V, total_kv * sizeof(float));
        cudaMalloc(&d_O, total_q  * sizeof(float));
        cudaMemcpy(d_Q, h_Q, total_q  * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, h_K, total_kv * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_V, total_kv * sizeof(float), cudaMemcpyHostToDevice);

        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 blocks(
            (d_head + TILE_SIZE - 1) / TILE_SIZE,
            (seq_q  + TILE_SIZE - 1) / TILE_SIZE,
            batch_size * num_heads
        );

        printf("=== Test 2: batch=%d heads=%d seq_q=%d seq_k=%d d_head=%d ===\n",
               batch_size, num_heads, seq_q, seq_k, d_head);
        printf("Grid: (%d, %d, %d)  Block: (%d, %d)\n",
               blocks.x, blocks.y, blocks.z, threads.x, threads.y);

        attention_v4_multihead<<<blocks, threads>>>(
            d_Q, d_K, d_V, d_O,
            batch_size, num_heads, seq_q, seq_k, d_head, scale
        );

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return 1;
        }

        cudaMemcpy(h_O, d_O, total_q * sizeof(float), cudaMemcpyDeviceToHost);

        // The CPU reference uses the same head_stride = seq_q * d_head for Q/O
        // but seq_k * d_head for K/V — pass seq_k explicitly so it navigates correctly
        multihead_attention_cpu(h_Q, h_K, h_V, h_ref,
                                batch_size, num_heads, seq_q, seq_k, d_head, scale);

        float err_val = max_abs_error(h_O, h_ref, total_q);
        printf("Max absolute error vs CPU reference: %e\n", err_val);
        printf("Result: %s\n\n", err_val < 1e-3f ? "PASS ✓" : "FAIL ✗");

        free(h_Q); free(h_K); free(h_V); free(h_O); free(h_ref);
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    }

    return 0;
}
