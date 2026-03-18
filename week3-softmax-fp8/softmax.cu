#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


/* 

---------------------------- PRELUDE/Accumulated Knowledge ---------------------------

My current understanding of softmax is as follows:

Step One: All values are raised to the power of e.
Step Two: Sum all values
Step Three: Divide each value by sum

The importance of each step is as follows (from my current mental model):

Step One:
 - We raise our value as a power so that we can transform negative values into positive values
 - We raise our value as a power so that if our said value is 0, we still return a value of 1. This is important as it takes away the possibility of dividing by 0 in Step Three.
 - We raise our values to the e specifically for the following reason:
    > During back propagation, where we tell each neuron its responsibility towards the incorrect value, we require taking partial derivatives of our rate of error.
       The derivative of e is e, this allows infomration to propogate more efficiently during back propagation. If we were to have each value raised to say 2, 
       the derivative of 2 is ln|2| and while this works, it is not as efficient as e.

Step Two:
 - We sum all the values so that in Step three we can divide each individual value so that we can find its contribution towards the sum.
   It essentially transforms any set of values to a 0 to 1 range, which we can interpret as a probability distribution.
   I believe this is used in LLMs to to determine the possibility of the next token.

Step Three:
 - We divide each value by the sum so that we can find the contribution of each value towards the sum. This allows us to find the probability distribution of the values.


 We do this once we have calculated or Query and Key DOT product, our softmax takes those row values (1 query, many keys) into a 0-1 probability representing
 how similar our individual key vector is to our query vector



---------------------------------------------------

A deeper dive into tree reductions

The core requirement for a tree reduction is said operation must be associative where the order in which we do said operation does not matter.

Example:
   - Addition:
       > 1 + 2 = 3
       > 2 + 1 = 3
    - Max
       > max([4,2,3]) = 4
       > max([2,3,4]) = 4

 If we ever need a single answer from many threads AND our operation is associative, tree reduction can be used

*/



/*

Roofline:
- Arithmetic Intensity: 4.29
   > We are memory bound, we cannot transfer memory fast enough to satsify cores
   > We have:
      - 3 Global memory reads
      - 2 Global memory stores

- Fails with larger numbers, NaN
*/
template <int BLOCK_SIZE>
__global__ void naive_softmax(const float* input_matrix, float* output_matrix, int num_cols) {
/**************************** Initialize row pointers ***************************************************** */

    int row = blockIdx.x; // One block per row
    const float* input_row  = input_matrix + row * num_cols; // Where does our input row begin for this block
    float*       output_row = output_matrix + row * num_cols; // Where does our output row begin for this block

/**************************** Pass 1: exp(input_row[col]) ***************************************************** */

    // We iterate through our input, we apply our e^value, and export to output row
    // We grid stride to ensure all our BLOCK_SIZE # of threads can handle BLOCK_SIZE < NUM_OF_COLS
    for (int col = threadIdx.x; col < num_cols; col += BLOCK_SIZE) {
        output_row[col] = expf(input_row[col]); // Global mem read + store
    }
    __syncthreads();


/**************************** Reduce: sum all exp values ***************************************************** */
    __shared__ float smem[BLOCK_SIZE];
    float partial_sum = 0.f;

    // We read our output row, and add to our thread's local sum
    // We grid stride to ensure all our BLOCK_SIZE # of threads can handle BLOCK_SIZE < NUM_OF_COLS
    for (int col = threadIdx.x; col < num_cols; col += BLOCK_SIZE) {
        partial_sum += output_row[col]; // Global mem read
    }

    // Add our thread's partial sum to its respective SMEM slot
    smem[threadIdx.x] = partial_sum;
    __syncthreads();

    // Tree reduction: halve the active threads each step until smem[0] holds the total
    // This is the same sequential-addressing reduction from Week 1
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }

/**************************** Pass 2: normalize ***************************************************** */
    // smem[0] now holds the total sum of all exp values for this row
    float row_exp_sum = smem[0];

    // Divide each stored exp value by the total sum to get probabilities
    for (int col = threadIdx.x; col < num_cols; col += BLOCK_SIZE) {
        output_row[col] /= row_exp_sum; // Global mem read + store
    }
}










/*
Stable Softmax:

The key insight into how we resolve the issue of NaN overflow when we have large numbers is the following.

Say we are using FP32. For FP32 any value above 88.7 when we do e^value will cause a overflow e.g. a NaN.

The trick with stable softmax is we subtract all of our values by the maximum number in our row.
This might sound odd at first, but the reasoning is quite sound. In softmax, what determines the final probabilities is the differences between values, not the values themselves

Lets see the following example of us applying this principle.
   - The difference between numbers [20,50] is 30
Now what if we subtract both numbers by a constant value 50, our maximum?
   - The difference between numbers [-30, 0] is 30


The same principle applies to our softmax problem where:
   - We solve our overflowing problem by subtracting all values by our maximum
   - The differences between all of our numbers remain the same since they were all subtracted by the same value.


*/
template <int BLOCK_SIZE>
__global__ void stable_softmax(const float* input_matrix, float* output_matrix, int num_cols) {

    int row = blockIdx.x;
    const float* input_row  = input_matrix  + row * num_cols;
    float*       output_row = output_matrix + row * num_cols;

    __shared__ float smem[BLOCK_SIZE];

    /**************************** Pass 0: find row max ****************************/

    //Read input, compare to max
    float thread_max = -INFINITY;
    for (int col = threadIdx.x; col < num_cols; col += BLOCK_SIZE) {
        thread_max = fmaxf(thread_max, input_row[col]); // Global mem read
    }

    smem[threadIdx.x] = thread_max;
    __syncthreads();

    //We use tree reduction to find maximum across warps
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    float row_max = smem[0];

    /**************************** Pass 1: exp(x - row_max) **********************/

    for (int col = threadIdx.x; col < num_cols; col += BLOCK_SIZE) {
        output_row[col] = expf(input_row[col] - row_max); // Global mem read + store
    }
    __syncthreads();

    /**************************** Reduce: sum all exp values ********************/

    float partial_sum = 0.f;
    for (int col = threadIdx.x; col < num_cols; col += BLOCK_SIZE) {
        partial_sum += output_row[col]; // Global mem read
    }

    smem[threadIdx.x] = partial_sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    /**************************** Pass 2: normalize *****************************/

    float row_exp_sum = smem[0];
    for (int col = threadIdx.x; col < num_cols; col += BLOCK_SIZE) {
        output_row[col] /= row_exp_sum; // Global mem read + store
    }
}


/*
Online Softmax

The fundamental of online softmax is that rather than doing multiple global reads, we have a running sum and max number to determine
our softmax.

We can view this similar to shared memory tiling where rather than needing to view our entire row at once, we essentially do a single
pass across our data where we modify our max sum and max num variables.

The way online softmax rescales its values while keeping a running sum is essentially the same principle we see in neural networks,
the Two Tree Framework, and in CUDA.

It uses the fundamental formula of 

A * B + C

or we can see it as

(Dimension A) * (Bridge from dimension A to B) + (Offset within dimension B)

Where say our max number is 5, but we come across the number 8, our new max. How softmax applies this formula is the following to
transform its old running sum to its new running sum is

(OldMaxValue) * (Bridge OldMaxValue to NewMaxValue) + (Offset in new value)
->
(Dimension A) * (The Difference between Dimension A and B) + (Offset within our new dimension B)

which translates to the actual formula for softmax when we come across a new maximum

(oldRunningTotal) * e^(oldMax - newMax) + e^(currentNum - newMax)

->

NOTE: exp(currentNum - newMax) will always be 1 because their both the same number, so we just use 1.f to avoid 
the extra computation


(FOR MORE INFORMATION ON THIS POINT OF VIEW AND HOW IT WAS DERVIED, SEE PARKING_LOT.MD)


*/
template <int BLOCK_SIZE>
__global__ void online_softmax(const float* input_matrix, float* output_matrix, int num_cols) {
/**************************** Initialize row pointers ***************************************************** */
    int row = blockIdx.x;
    const float* input_row  = input_matrix  + row * num_cols;
    float*       output_row = output_matrix + row * num_cols;

    __shared__ float smem_max[BLOCK_SIZE];
    __shared__ float smem_sum[BLOCK_SIZE];

    float max_number = -INFINITY;
    float runningTotal = 0.f;
/****************************Get our local running sum and max number***************************************************** */
    //Grid stride, allow our threads to scale to our problem size
    //This loop is about getitng our running sum, and max number
    for (int col = threadIdx.x; col < num_cols; col += BLOCK_SIZE) {
        //Get our current number
        float current_number = input_row[col]; //Global read

        if (current_number > max_number) {
            //If our new max, convert our old running sum to our new dimension using A * B + C
            runningTotal = runningTotal * expf(max_number - current_number) + 1.f;
            //Set our new max
            max_number = current_number;
        } else {
            //No need to rescale, just apply our running sum 
            runningTotal += expf(current_number - max_number);
        }
    }

    //Each thread stores both its running sum and max in its respective SM bank
    smem_max[threadIdx.x] = max_number;
    smem_sum[threadIdx.x] = runningTotal;
    __syncthreads();
/**************************** Aggregate our running sum and max across our block ***************************************************** */
    //Tree reduction, we use this to aggregate results across the block into a single number
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        //Ensure we dont go out of bounds with our grid stride
        if (threadIdx.x < stride) {
            //Select our maximum, and the maximum to collapse
            float m_a = smem_max[threadIdx.x];
            float m_b = smem_max[threadIdx.x + stride];

            //Select our running sum, and the running sum to collapse
            float d_a = smem_sum[threadIdx.x];
            float d_b = smem_sum[threadIdx.x + stride];

            //If our max is bigger than our max to collapse
            if (m_a >= m_b) {
                //If true,
                // We convert our running sum to collapse to our orginial running sum
                //                  Dimension B * Bridge from dimension B to A + add A offset
                smem_sum[threadIdx.x] = d_b * expf(m_b - m_a) + d_a;
            } else {
                //If false,
                // Set our current threads location's max to our maximum to collapse
                smem_max[threadIdx.x] = m_b;
                // We convert our current running sum (dimension A) to our new running sum (dimension B)
                //                  Dimension A * Bridge from dimension A to B, add B offset
                smem_sum[threadIdx.x] = d_a * expf(m_a - m_b) + d_b;
            }
        }
        __syncthreads();
    }

/****************************Calculate each values softmax value, and store***************************************************** */

    //Grab our aggregated max and running sum
    float row_max = smem_max[0];
    float row_sum = smem_sum[0];

    //Divide each value with our running sum and subtract with max number to avoid overflow
    for (int col = threadIdx.x; col < num_cols; col += BLOCK_SIZE) {
        output_row[col] = expf(input_row[col] - row_max) / row_sum; //Global read and store
    }
}












// NOTE THE CODE BELOW IS AI GENERATED








/**************************** Helpers ***************************************************** */
static void fill_rand(float* buffer, int num_elements, float range_lo, float range_hi) {
    for (int i = 0; i < num_elements; i++) {
        buffer[i] = range_lo + (range_hi - range_lo) * (float)rand() / RAND_MAX;
    }
}

// CPU reference implementation — numerically stable (subtracts row max before exp)
// This is our ground truth, same role as the CPU triple-loop in the GEMM main
static void cpu_stable_softmax(const float* input_matrix, float* output_matrix, int num_rows, int num_cols) {
    for (int row = 0; row < num_rows; row++) {
        const float* input_row  = input_matrix  + row * num_cols;
        float*       output_row = output_matrix + row * num_cols;
        float row_max = -1e30f;
        for (int col = 0; col < num_cols; col++) {
            row_max = fmaxf(row_max, input_row[col]);
        }
        float row_exp_sum = 0.f;
        for (int col = 0; col < num_cols; col++) {
            output_row[col] = expf(input_row[col] - row_max);
            row_exp_sum += output_row[col];
        }
        for (int col = 0; col < num_cols; col++) {
            output_row[col] /= row_exp_sum;
        }
    }
}




/**************************** Main ***************************************************** */
int main() {
    const int BLOCK_SIZE = 256;
    srand(42);

    const int NUM_ROWS = 512;
    const int NUM_COLS = 1024;
    const int TOTAL    = NUM_ROWS * NUM_COLS;

    // Host allocations
    float* h_input  = (float*)malloc(TOTAL * sizeof(float));
    float* h_ref    = (float*)malloc(TOTAL * sizeof(float));
    float* h_output = (float*)malloc(TOTAL * sizeof(float));

    // Device allocations
    float *d_input, *d_output;
    cudaMalloc(&d_input,  TOTAL * sizeof(float));
    cudaMalloc(&d_output, TOTAL * sizeof(float));

    // Reusable verify lambda — same pattern as your GEMM verify
    auto verify = [&](const char* name) {
        cudaDeviceSynchronize();
        cudaMemcpy(h_output, d_output, TOTAL * sizeof(float), cudaMemcpyDeviceToHost);

        float max_err   = 0.f;
        int   nan_count = 0;
        for (int i = 0; i < TOTAL; i++) {
            if (isnan(h_output[i])) { nan_count++; continue; }
            max_err = fmaxf(max_err, fabsf(h_ref[i] - h_output[i]));
        }

        if (nan_count > 0)
            printf("%s: NaN count %d / %d\n", name, nan_count, TOTAL);
        else
            printf("%s: %s (max error: %.2e)\n", name,
                   (max_err < 1e-5f) ? "CORRECT" : "INCORRECT", max_err);
    };

    // ── Test 1: normal input [-3, 3] ──────────────────────────────────────
    // Naive softmax should handle this cleanly — values are well within float range
    printf("\n=== normal input [-3, 3] | rows=%d cols=%d ===\n", NUM_ROWS, NUM_COLS);
    fill_rand(h_input, TOTAL, -3.f, 3.f);
    cpu_stable_softmax(h_input, h_ref, NUM_ROWS, NUM_COLS);
    cudaMemcpy(d_input, h_input, TOTAL * sizeof(float), cudaMemcpyHostToDevice);

    naive_softmax<BLOCK_SIZE><<<NUM_ROWS, BLOCK_SIZE>>>(d_input, d_output, NUM_COLS);
    verify("naive_softmax");

    // ── Test 2: large input [990, 1010] ───────────────────────────────────
    // expf(1000) overflows to inf → inf/inf = NaN
    // Numerically stable softmax (coming next) will fix this
    printf("\n=== large input [990, 1010] | rows=%d cols=%d ===\n", NUM_ROWS, NUM_COLS);
    fill_rand(h_input, TOTAL, 990.f, 1010.f);
    cpu_stable_softmax(h_input, h_ref, NUM_ROWS, NUM_COLS);
    cudaMemcpy(d_input, h_input, TOTAL * sizeof(float), cudaMemcpyHostToDevice);

    naive_softmax<BLOCK_SIZE><<<NUM_ROWS, BLOCK_SIZE>>>(d_input, d_output, NUM_COLS);
    verify("naive_softmax");

    // ── Test 3: stable softmax on normal input ────────────────────────────
    printf("\n=== stable_softmax | normal input [-3, 3] ===\n");
    fill_rand(h_input, TOTAL, -3.f, 3.f);
    cpu_stable_softmax(h_input, h_ref, NUM_ROWS, NUM_COLS);
    cudaMemcpy(d_input, h_input, TOTAL * sizeof(float), cudaMemcpyHostToDevice);
    stable_softmax<BLOCK_SIZE><<<NUM_ROWS, BLOCK_SIZE>>>(d_input, d_output, NUM_COLS);
    verify("stable_softmax");

    // ── Test 4: stable softmax on large input — should NOT produce NaN ────
    printf("\n=== stable_softmax | large input [990, 1010] ===\n");
    fill_rand(h_input, TOTAL, 990.f, 1010.f);
    cpu_stable_softmax(h_input, h_ref, NUM_ROWS, NUM_COLS);
    cudaMemcpy(d_input, h_input, TOTAL * sizeof(float), cudaMemcpyHostToDevice);
    stable_softmax<BLOCK_SIZE><<<NUM_ROWS, BLOCK_SIZE>>>(d_input, d_output, NUM_COLS);
    verify("stable_softmax");

    // ── Test 5: online softmax on normal input ────────────────────────────
    printf("\n=== online_softmax | normal input [-3, 3] ===\n");
    fill_rand(h_input, TOTAL, -3.f, 3.f);
    cpu_stable_softmax(h_input, h_ref, NUM_ROWS, NUM_COLS);
    cudaMemcpy(d_input, h_input, TOTAL * sizeof(float), cudaMemcpyHostToDevice);
    online_softmax<BLOCK_SIZE><<<NUM_ROWS, BLOCK_SIZE>>>(d_input, d_output, NUM_COLS);
    verify("online_softmax");

    // ── Test 6: online softmax on large input — the real correctness gate ─
    printf("\n=== online_softmax | large input [990, 1010] ===\n");
    fill_rand(h_input, TOTAL, 990.f, 1010.f);
    cpu_stable_softmax(h_input, h_ref, NUM_ROWS, NUM_COLS);
    cudaMemcpy(d_input, h_input, TOTAL * sizeof(float), cudaMemcpyHostToDevice);
    online_softmax<BLOCK_SIZE><<<NUM_ROWS, BLOCK_SIZE>>>(d_input, d_output, NUM_COLS);
    verify("online_softmax");


    free(h_input); free(h_ref); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    return 0;
}


/**************************** PyTorch extension wrapper ***************************************************** */
// Everything below this line is only used when building via build.py (Python/PyTorch path)
// It lets test_softmax.py call your kernel directly and compare against F.softmax

#ifdef WITH_TORCH
#include <torch/extension.h>

torch::Tensor softmax_forward(torch::Tensor input) {
    int num_rows = input.size(0);
    int num_cols = input.size(1);

    torch::Tensor output = torch::empty_like(input);

    constexpr int BLOCK_SIZE = 256;
    int smem_bytes = BLOCK_SIZE * sizeof(float);

    naive_softmax<BLOCK_SIZE><<<num_rows, BLOCK_SIZE, smem_bytes>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_cols
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softmax_forward, "Naive softmax CUDA");
}
#endif