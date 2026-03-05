Shared Reduced Sum:
https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

The following talks about 7 increasingly optimized versions of Shared Reduced Sum!
I was able to implement the first three, but this would be a lovely optimization opportunity.
>Credits to Mark Harris for this amazing insight!

- Atomic add can be used to avoid consecutive kernel launches.
Pro: Great for small n sizses
Con: For large n sizes, serialization can cause deficit in performance.
>Overall requires more experimentations to conclude behavior.



Matrix Transpose:
https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/

A very helpful source!
> Also credits to Mark Harris for this amazing insight!


GEMM:
https://nerffer.github.io/matrixMultiplicationVisualization/
http://matrixmultiplication.xyz/

Both of the following links are great visualizations for matrix multiplication

-----------------------
Why is the naive uncoalesced access. At first I thought it was because in Matrix B we request for data column wise.
Although one could argue on a warp scale, Matrix A actually causes our uncoalesced access where on a warp scale, we are going down each row where
the addresses are not consecutive.

                *Disputed Self Theory, to be parked in "PARKING_LOT.MD" for after project submission understanding*
 At first I assumed Matrix B was the reason for this uncoalesced access, since on a thread level, we are requesting data from multiple rows.
 In reality on a warp scale, Matrix A is the reason why we may have uncoalesced access.
 This is because while a single thread in Matrix A travels across a row (coalesced), on a warp scale of 32 threads, we are actually 
 following the row dimension (threadIdx.y). This means:
   - Thread 0: Requests for address 0
   - Thread 1: Requests for address 33 because we are going down the Matrix A rows

---> Future perspective

Matrix A requests 32 bytes, but they are not used instantly, and have to wait for the next k iteration use the requested data.
This leaves a chance for the L1 or L2 to evict the data before the kernel has a chance to use it

Matrix B requests 32 bytes and is not reliant on k. It is reliant on the iterator value col. This col value is determined by threadIdx.x
which means all 32 threads can simutaniously use the bytes requested rather than needing to wait on iterator k.


--------------------




Interesting topics:
--------------------
- When you come across a row-wise matrix, you must use row wise access form of base formula

Coordinate * Stride + Offset !!

This is because when the array is flattened (how the kernel sees it), row numbers are kept contingent.

If we used column wise access formula, we would get the wrong data!!

(This insight took many hours I fear)
--------------------



--------------------

Flattening Row Wise Matrix:
WhichRow * HowBigIsTheRow + WhereInTheRow (column)

Flattening Column Wise Matrix:
WhichColumn * ColumnSize + WhereInColumn (row)


Single axis Coordinate Access:
Where * Size + WhereIn
 or
(higher level) * (lower level) + (lower level offset)
 
--------------------


--------------------
If threads are to small for problem size (Threads to Shared Memory), use Flatten > Expand > Fold
Assume Threads 8x8
Assume Shared Memory 32x32

1. Flatten
     - You flatten threads to 0-63 ()
2. Expand
     - You scale those 0-63 threads to match 1024 
3. Fold
     - You use divison 32 for row, modulus 32 for column


Lock + Slide + Micro
1. #TODO
--------------------


--------------------
The theoretical framework that came from hours of struggling, I believe it will help me with my current GEMM Register Accumlated 2D, and the kernels in the future.

This framework captures my current scope of knowledge, and I'm sure it doesn't apply to all kernels, and also may be inconsistent.

 > The Two Tree Framework <

      
Two Tree Table:

  Execution (Threads)       Memory (Data)

Grid   32 x 32          Global   1024 x 1024
Block   8 x 8           Shared    32 x 32
Thread    1             Register   4 x 4

- This tree table allows us to explicitly see the difference between the Execution and Memory trees.
- This tree table shows us whether we need a for loop for our Execution to Memory mapping
   > Ex. For our 32 x 32 Grid to fully cover our 1024 x 1024, we must slides across Global 32 times (1024/32).
   > Ex. For our 8 x 8 Block to fully map to our 32 x 32 Shared Memory, we must stamp across Shared Memory 16 times (32*32/8*8)
   > Ex. For our 1 thread to fully map to our 4 x 4 register, our thread must iterate 16 times over our Register (16/1) 
- This tree also allows us to confirm whether our variable correctly covers the respective dimension space
   > Ex. Say we have
      - int aCol = tileId * TILE_SIZE + sCol;
    > Assuming aCol is in the Matrix A dimension (Global) we can say that our maximum value must be 1023.
      - tileId = 0-31
      - TILE_SIZE = 32
      - sCol = 0-31
    > When we take the maximum value of each variable, we get at total of 1023. This means we have properly covered this 1024 dimension space.

Two Tree Dimensional Analysis Table

EXECUTION TABLE

 Level |   Size   |   Size Unit     |          Index           |     Unit      |
-------|----------|-----------------|--------------------------|---------------|
Grid   | gridDim  |    [BLOCKS]     | blockIdx.x   blockIdx.y  |    [BLOCK]    |
Block  | blockDim | [THREADS/BLOCK] | threadIdx.x  threadIdx.y |   [THREAD ]   |
Thread |    1     |    [THREAD]     | localId                  | [FLAT_THREAD] |


MEMORY TABLE

 Level   |     Size        |    Size Unit      |    Index    |   Unit    |
---------|-----------------|-------------------|-------------|-----------|
Global   |    M, N, K      |    [ELEMENTS]     | aCol   aRow | [ELEMENT] |
         |                 |                   | bCol   bRow | [ELEMENT] |
         |                 |                   | cCol   cRow | [ELEMENT] |
         |                 |                   |             |           |
Shared   |   TILE_SIZE     | [ELEMENTS/BLOCK]  | sCol   sRow | [ELEMENT] |
Register | WORK_PER_THREAD | [ELEMENTS/THREAD] | rCol   rRow | [ELEMENT] |


Two Step Verification:
- We use dimensional analysis to ensure we have the correct units
- We use Max Index Boundary Check to make sure the calculated value is within the range we expect

- The Size and Size Unit allow us to bridge our Execution and Memory Table
- When we:
   > Flatten from 2D to 1D, we can use the following formula  WhichRow * SizeOfRow + WhereInRow
   - We can use dimensional analysis to confirm whether our units are consistent.
- 