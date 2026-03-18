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






---------------------------------------------------------------------------------------------------------

Can we understand our answers natural dimension, and how we can derive our tile size, register dimension before we code??

Potential framework

<img width="2160" height="1620" alt="image" src="https://github.com/user-attachments/assets/d8abef46-a993-4afb-809d-167d546c3dea" />

<img width="2160" height="1620" alt="image" src="https://github.com/user-attachments/assets/61577395-7bec-49a9-b8a8-a8431a8db21c" />

<img width="2160" height="1620" alt="image" src="https://github.com/user-attachments/assets/f8c8de69-0406-493b-a1fb-dc2dec903601" />


<img width="2160" height="1620" alt="image" src="https://github.com/user-attachments/assets/43ed31dd-cb92-4b19-a2f8-8d58934130f3" />

<img width="2160" height="1620" alt="image" src="https://github.com/user-attachments/assets/95ee304f-4f88-42bd-afaf-d688b4f0834f" />

<img width="2160" height="1620" alt="image" src="https://github.com/user-attachments/assets/1b8de2ce-f5da-42aa-bd52-15e6aa706abe" />

<img width="2160" height="1620" alt="image" src="https://github.com/user-attachments/assets/8bcfec9d-d5fc-4c7a-b0a8-5df8745bca13" />

<img width="2160" height="1620" alt="image" src="https://github.com/user-attachments/assets/cd748bca-2f0a-4e40-ac68-0c038ce34b14" />

------------------------------------------------------------

------------------------------------------------------------

SOFTMAX INSIGHT

Online softmax is insightful. As I learn more and more techniques in the world of computer science,
its seems the most impactful improvements are not ones that require significants amount of code, but rather
they approach the problem from a different angle. Where rather coding the same thing with more lines/complexity to improve on edge cases
for minor improvement, it can prove very worthy to step back from the code itself, and approach the problem as something to be solved
at the fundamental level, rather than just by code.


One thing I noticed from softmax is the formula to rescale an old running sum is

oldSum * (oldSum-newSum) + newSumAddition

I keep seeing this fundamental formula of 

A * B + C

I see it in neural networks where
Weight * Input + Bias
We use it to calculate how much our input is effected by the weight

I see it where for the whole Two Tree framework, the whole idea was
Coordinate * Stride + Offset
To bridge our dimensional hierarchy from execution to memory


This formula seems so fundamental and monumental, but so simple at the same time. My current understanding of it is essentially,
no matter the subject whether that be neural networks, the Two Tree Framework or CUDA work it comes down to

(Dimension A) * (Bridge from Dimension A to B) + Offset within B dimension


But the thing is, this formula is recursive, we can bridge many dimensions at once! We can have the following:

(Dimension A) * (Bridge from Dimension A to B) + (Dimension B) * (Bridge from Dimension B to C) + Offset within C dimension


The question is, why is this equation so fundamental, why is bridging dimensions so a crucial necessity where every modern GPU
optimizes the fused multiply add formula? I understand its used in neural networks and CUDA and everything inbetween, but the question is why
this specific formula.

I assume its because of its simplicity. I assume its because its the simplest way to express... something... maybe its the simplest bridge from
Dimension A to Dimension B?
#TODO ^^^^^^^^^^^^^^^^^^^^^^^^^^

Either way, in the shadow framework, we treat our Register, Shared and Global as different dimensions because say the following

Register has a bit address of

1234

But Shared memory has a bit address of

123456

These additional two digits are a way to express more information, which we can see as a new dimension, especially when we see it as

1234 + 56

Where say 1234 are the same numbers, and 56 are the values that can change, its a way to express more information. To help me view it,
I see it as the "shadow" casted from the register to the shared memory, where the shadow is the possible new values we can express within our new
5 and 6 bits.

Knowing this, we can use our A * B + C to bridge these dimensions, infact I assume that is how its used already. The thing is, we can extend this from

Register > Shared > Global,

but how does the execution tree come into this. Its our job to map our execution tree as well as we can to our memory tree.

The question is, given a problem, if we can derive its natural dimension within the register shared or global, or just the natural dimension of the answer itself
if we can match the same geometric structure the memory hierarchy creates with our execution hierarchy, does that mean we can pre-plan our CUDA code not from the
perspective of coding to tackle the problem, but rather forming our code the dimensions of the problem?

Rather to be more specific, if we can understand the natural dimensions of our problem, can the structure of our code/execution hierarchy come about in a way
that can be systemically derived?



------

DEFERRED, SHOULD DO

FP8 GEMV 

PRACTICE FP8 INPUT TO FP32 ACCUMULATE, BACK TO FP8

Essentially, we do computation in FP32 as it allows for more expression in our value


------------

SOFTMAX:

Two Tree Framework does not capture tree reductions, or 1D vectors, need to update afterwords