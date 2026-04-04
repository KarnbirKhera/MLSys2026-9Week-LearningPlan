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

Two Tree Frameework is too rigid for the flow dense attention, Load, Compute and Store does not capature it

FSMTTF

Is a for loop a riemann sums across a given curve or line??

Our index is the x value, our given value is the y.

Is every for loop essentially mapping a curve?

Is two for loops mapping a two dimensional space of that curve? Maybe like the gradient descent visual we get a idea of our shape using two loops? Maybe it maps our problem space?

To test, we need to apply this mindset to 4x4 registered allocated GEMM where does it make sense?

--->

If a for loop can be a riemann sum that covers a line,

two loops covers an area

and three loops covers an

Area + Volume

This volume may be confusing but we know it might be true because loook at the formula for say

a square to a cube

A square is B * W

A cube is B * W * H

Lets apply the same thinking to a circle to a cylinder

the formula for a circle is pi * r^2

but the formula for a cylinder is pi * r^2 * h

but the funny thing is in CUDA

To obtain a single point from our execution hierarchy to memory hierarchy, we have to do

WhichBlockAmI * HowBigIsThisBlock + WhereInThis

Which translates to

Coordinate * Stride + Offset

BUT THE TENSOR CORE FORMULA IS

(COORDINATE * STRIDE + OFFSET ) * VOLUME?

This might seem confusing so lets use the example you said for how 3D convolutions work

we have

coordinate * stride + offset * time

and how that can relate to tensor cores is that this 3rd dimension is TIME

Where

say a 16x16 is we're processed 16 x16 data

but 16x16x16 is were processed 16 TIMES of that 16x16 data AT ONCE

If we see this geometircally,

we go from not just the 2d plane, but its VOLUME/TIME

ARE TENSOR CORES ARE JUST ONE BIG PRAGMA UNROLL?


------------------------------------------------------

SOFTMAX:

Two Tree Framework does not capture tree reductions, or 1D vectors, need to update afterwords

Does not capture tensor cores as they work in a different dimension. But its like this most likely

Linear Regression: Complex patterns are seen as non-linear lines
Polynomial Regiressoin: The same complex pattern when viewed from a higher dimension is linear

To extend the framework, we must view these complex patterns (Tensor cores) from a higher dimension.

The challenge would be to bridge seamlessly between this low dimension (CUDA Core) and high dimension (Tensor Core)
and I suspect we might be able to use A * B + C again.






-------------------------
Dense Attention:

Multi-Headed Attention
    We incorporate a new dimension Z within our block. I am new to this so this will be interesting none the less.
    It seems each head does not require communication among one another.......

    Tensor Core is essentially a unrolling the loop on the third dimension. Earlier in my parking lot I mentioned that the 
    3rd dimension might not just be a spatial dimension like the first two, but rather a dimension of time. I say this because when I look at 3D convolutions,
    The first two for loops give us the pixels x and y, but the third for loop gives us seperate frames, so a time dimension if you will.

    I made a LinkedIn post about this and why I think this at the moment https://www.linkedin.com/feed/update/urn:li:activity:7440774158847913984/?originTrackingId=bfJ0MRzY8tLPNzV%2FDZiFgg%3D%3D

    The thing is, if this is true, that means that this use of our 3rd dimension is essentially us just doing the same attention we did in the kernel before, but just the size of our Z dimension at once.

    Again, if this is true, that means Tensor Cores were quite literally made for this, or rather, multi headed attention was quite literally made for Tensor Cores!!
 




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
    
    int batch_idx = blockIdx.z / num_heads;
    int head_idx  = blockIdx.z % num_heads;


    ---------------------

    When dealing with non-communicative operations such as addition or substraction, we can elevate them to e^ . This means we can 
    use it as a running sum like softmax does

    -----------------------



   The external PyTorch test taught me that I had assumed both our passed Q K V matrices we're all square. 

    This shows a gap in my kernel process where I should first write down my assumptions, so that they can be tackled through edge case handling.

    Knowing this, I cannot assume that Q K and V will be proper square matrices, and my code should handle such cases

    DIMENSIONS OF OUR INPUT
    Q -> (seq_q,d_head)
    K -> (seq_k, d_head)
    V -> (seq_k, d_head)
    O -> (seq_q, d_head)




    -----------------------

    RoPE

    Rope works by rotating our vectors by x angle. This allows our attention to compare the RELATIVE difference between our vectors.

    This works better than fixed positioning where say position one gets added to vector one, position two gets added to vector 2 etc.

    RoPE allows us to deconstruct our K matrice into two parts, the keys themselves, and the positions. This allows us to compress K (no positioning) and V into 
    a single latent compresison to later be upscaled.

    The interesting thread I keep seeing is when comparing two objects, its not the absolute position that matters, but rather the relative difference between them
    I saw this in soft max where we were able to subtract all of our values by the max number because we were not measuring the absolute position, but rather its
    relative position to other values.

    This is interesting and I'm curious to see where this relative difference can be applied to past kernels where the assumption may have been made we require
    the absolute position rather than just the relative.



    ------------------------------------------

    Paged KV Multi Latent Attention
    
    TTF needs a model for paged two phase addressing and tensors with two roles, for an example our ckv_cache has both compressed k and v



    -----------------------------

    If given an input tensor, and an output tensor


    Each of the dimensions in the output tensor must be parallized, these are our output dimensions

    If a dimension is in our input tensor, but not in our output tensor, that means it was reduced/summed up.
    This reduction/summing is done by a for loop, rather than blockIdx.x, blockIdx.y or blockId.z


    Every part of our execution hierarchy is PARALLELIZED just in different COARSES/FINENESS...

    blockIdx = COARSER 
    threadIdx = FINER

    As we know threadIdx.x is in lockstep


    AT A FUNDAMENTAL LEVEL WE ARE DOING

    EXECUTION TENSOR _DOT PRODUCT_ MEMORY TENSOR 
     -> SHARING DIMENSION GETS REDUCED, OUR BRIDGE
     -> OUTPUT DIMENSIONS TELL US HOW RELEVANT EACH VECTOR IS WHERE HOW GEOMETRICALLY ALIGNED IS OUR EXECUTION VECTOR TO OUR MEMORY VECTOR



    ALL DATA CAN BE REPRESENTED IN A GEOMETRICAL SPACE, HENCE ANY AND ALL KERNELS CAN BE REPRESENTED IN A GEOMETRIC SPACE. THUS ALL KERNELS MUST BE SOME MODIFCATION OF OUR PROBLEM SPACE (DATA).

    DOT PRODUCT = REQURIES SAME DIMENSION, HOW SIMILAR ARE OUR VECTORS IN OUTPUT DIMENSIONS. MAPS EXECUTION TO MEMORY BUT HOW/WHAT DOES IT MEAN


    ----------

    IF WE GO FROM A LOAD TO A COMPUTE, THE LOAD N EEDS A SYNC THREADS IN THE END, THE COMPUTE NEEDS A SYNC THREADS IN THE OUTER FOR LOOP

    IF ITS A COMPUTE -> COMPUTE WE REQUIRE A SYNC THREADS WITHINTN HE FOR LOOP


    FUNDAMENTALLY, WE N EED A SYNC THREADS WHEN WE TRANSITION FROM MOVING MEMORY TO WORKING WITH IT

    AND COMPUTE COMPUTE

    SO IF

    COMPUTE ALWAYS HAS  SYNC THREADS BECAUSE EACH COMPUTATION IS DEPENDENT ON THE LAST

    BUT IF A LOAD GOES ITNOA  COMPUTE, WE NEED A SYNC THREADS TO ENSURE OUR DATA HAS ARRIVED BEFORE BEING PROCESSED

    BUT A LOAD LOAD IS FINE AS EACH DATA IS NOT DEPDENDENT ON THE PRIOR



    RULE: THE OPERATION BEFORE NEEDS TO BE FULLY COMPLETED, MUST SYNC THREADS, AND WITHIN COMPUTATION FOR LOOPS. 

    BASICALLY THE GEOMETRY DIMENSION MUST BE FULLY VISIBLE BEFORE PROCEEDING

    

    IN DOT PRODUCT IN TWO TREE,

    WE HAD A SLIDE THAT WAS OUR REDUCTION DIMENSION, OUR AREA WAS OUR OUTPUT DIMENSION

    THE ONE FOR LOOP (SLIDE) IS US COVERING A SINGLE DIMENSION WITHIN OUR TENSOR, THE REDUCTION DIMENSION

    THE TWO FOR LOOP (AREA) IS US COMPARING TWO DIMENSIONS AT ONCE!!! THE OUTPUT DIMENSION



    A FOR LOOP -> SINGLE DIMENSION OF OUR TENSOR COVERED
    2 FOR LOOPS -> TWO DIMENSIONS OF OUR TENSOR COVERED 
    3 FOR LOOPS -> ALL THREE DIMENSOINS OF OUR TENSOR COVERED

    REMEMBR THAT THESE DIMENSIONS OVERLAP ONE ANOTHER

    -------------------------------------------------------

KV Cache Paging

This follows similar concept I mentioned on LinkedIn 

LinkedIn: https://www.linkedin.com/feed/update/urn:li:activity:7442586118803496960/

(I may have mentioned in here as well) but the idea is as follows.

For Multi-Headed attention the big difference was that we had the following input tensor structure (example)

[seq_q, d_head, batch_size, num_head]

From this structure we see that this is a 4D tensor. The problem with a 4D tensor is our GPU archiecture is made to at maximum
hold a 3 dimensional structure to process. We see this in its design where our execution hierarchy is

Grid
Block
Thread

And our memory hierarchy is

Global
Shared
Register

One can assume this is the case because at a hardware level I assume we arrive at diminishing returns when it comes to price per compute, as well as the 
physical space on the actual silicon board which prevents us from efficiently capturing a 4D tensor to process.

Because of this, multi-headed attention has a very clever work around. Instead of having a basic 4D tensor of 

[seq_q, d_head, batch_size, num_head]

we can actually flatten batch_size and num_head into a single dimension. This makes sense in principle since technically everything at the hardware level
is flat, and the concept of dimensions is a principle for us to understand how to percieve this hardware 1D world. This means after flattening batch_size and batch_head

we get a 3D tensor of

[seq_q,d_head,batch_size/num_head]

The question becomes how do we traverse this flattened dimension to get the information we need? From our prior knowledge we know the following assumption has thus far
been true.

To go from a 2D logical address to a 1D hardware address we use the following bridge.

Index * Stride + Offset

where for every Index * Stride, we are esesentially flatening the dimension, and we traverse the next dimension offset.
Now to go from a 1D hardware address (our 3rd dimensionn is essentially a flattened hardware address), we can use the following assumption:

Index / Stride
Index % Stride

Where index is our current location, and stride is how big each of our stored value is.

The divison gives us the row within this 1D hardware dimension
The modulus gives us the column within this 1D hardware dimension

This essentially allows us to treat our single flattened dimension as a two dimensional space, while maintaing the 3D tensor hardware constraint.



Now this comes with some downsides, which I will infer so if I was a reader I'd take caution in assuming if the following is actual based on hard truth.

We have to store our row and columnn values, this means a very slight increase in register pressure.

We can assume because we have two dimensions worth of data in a single dimension, compared to other dimensions that only hold
1 "meaning" if you will, one can imagine we may hit the "ceiling" limit on the number of bits we can actively store within our shared memory or register.
Although, this exact notation is countered by the fundamental idea of tiling, which says no matter how big our data is, if we process it piece by piece,
we can process the entire dataset.


At the moment, those are the cons I can think of. Now for the positives.

This allows us to store more information within dimension.

This also allows us to store information non-contingously. Where before for the formula of Index * Stride + Offset,
one could assume we need a uniform understaniding of everything before this Index * Stride has relevant and valid data. But if we use paging
We are directly accessing said stored data, without the assumption that everything before or after this is valid data. I believe this is the main point of paging
where for LLMs say we have 10 tokens to process, we dont want to allocate 2048 tokens worth of space, paging allows to allocate just the amount needed since our 
data doesnt ncessarily have to be stored next to each other. Although I'd admit at the moment the actual concept is rather unclear to me, so it definently needs some more
learning and refining.


Now for the inferred postivites, or the fun hypotheticals. 

If our limit is a 3D tensor, and within that 3D tensor we can flatten dimensions.
Can we theoretically flatten many dimensions within this 3D tensor, and if we can what is the diminishing return point, and at what point is
it considered valuable

-> Upon further thought, this means signficantly more register pressure, if our kernel is primarily compute bound, we can trade more register pressure for
occupancy. Although yes we can transfer more information per tensor, but that really isn't the limiting factor since our DRAM can hold a signficant amount of data.

-> Also, at I first I had thought that since we recieve more information per tensor, that meant we can do more calculations per input, but being able to express more
information per tensor still means the data has to travel from the DRAM to our actual kernel, so there really isn't  a significant point


Another fun theory is what if in our 3D tensor we have our first dimension soley as a paged index dimension. Where if we can to access say 
our third dimension at a specific spot, our 1st dimension can tell us where to find said information as it provides the page and column indice in the 3rd dimension.

This would allow us to more effectively use our memory. I say this because if KV caching is more efficient when it comes to memory allocation because of paging since it
no longer requires contingous memory storage, one could assume the same beneift remains if we were to page all three tensor dimensions given to us.

 -> This is rather interesting, and the pros are we not longer need to store data contingiously... hmm the problem might be though when we access said information, if it isnt
 stored next to each other on the registers, that means we lose parallezation benefit of our threadIdx.x which is the pinacle of GPU programming. None the less, it is a very interesting
 thought.


 

 TOPK INDEXER


     ///What does each thread contribute to in our paged diemnsion grid strided!!
    // This is essentially pre-mapping our thread to our paged dimension
    //Its like a piano where the keys are the threads, and they map to outer dimensions.
    // It doenst cost to traverse this outer dimension because the keys themselves are tied to them
    for (int i = threadIdx.x; i < NUM_HEADS * HEAD_DIM; i += blockDim.x) {
        int h = i / HEAD_DIM;
        int d = i % HEAD_DIM;
        smem_q[h * SMEM_STRIDE + d] = q_base[i];
    }


THIS IS QUITE LITERALLY WHAT WE THEORIZED EARLIER WITH KV CACHE, AND IT ACTUALLY TURNS OUT TO BE USED IN TOPK INDEXING?????????????!!!!!

"
This would allow us to more effectively use our memory. I say this because if KV caching is more efficient when it comes to memory allocation because of paging since it
no longer requires contingous memory storage, one could assume the same beneift remains if we were to page all three tensor dimensions given to us.

 -> This is rather interesting, and the pros are we not longer need to store data contingiously... hmm the problem might be though when we access said information, if it isnt
 stored next to each other on the registers, that means we lose parallezation benefit of our threadIdx.x which is the pinacle of GPU programming. None the less, it is a very interesting
 thought.
"



The Two Tree assumes continous uniform memoory access through Coordinate * Stride + offset, but we should lift this assumption to support paged access (2 dimensions flattened dimension).
>>>>>>>>>Can we geometrically understand our kernel to when paging is needed/more efficient than Coordinate * stride + offset access?


This framework will continue to grow, as more and more assumptions get challenged, the framework will be able to essentially capture a pattern, add it to its framework, allowing it to
express a new class of kernels. (Where we treat every kernel that breaks the framework as a Pokemon to be caught, or a structual pattern to capture, allowing our framework to express even more kernels) 


If a specific geometry keeps repeating, and the same load compute phase combinations keeps appearing, can we infer the load compute store based off the geometry alone.
Also if the geometry is the foundation of it, and if the load compute store pattern is the same for the same geometry, can we systematically derive the needed optimziations
for the geometry GIVEN the hardware constriants. Can this also tell us where our hardware is constrianed and cannot express our geometry, meaning can it tell us what 
specific hardware changes need to be done to express x class of kernels?



Mayhaps an exciting summer of hunting kernels that break the framework, a pokemon to be caught.
While literature will help, I find the active part of finding "pokemons" that break the frameworks assumption as a game to played and its one that I enjoy



Paging is needed if the tensor is 4D or above and needs to be flattened to 3D to match hardware, we can likely derive this need in the Geometry phase.



Once the framework matures, the next step would be to see if optimizations can be predicted given our current three layer structure, and whether a new structure may need to
be theorized. I say this is possible because just as code can only exist if it captures a repeatable pattern, an optimization must work if the conditions required to 
apply said optimzation are present. This repeatable pattern must be demonstrated somewhere, in some way, or dimension, even if the current 3 layer structure doesnt.



---------

If in a Multi GPU setup, can we individualize the functions that make up our kernel, and optimize the entire GPU/partial kernel around this?
Can this approach be useful for specific types of kernels compared to the naive? Can the Finite State Machine nature of the framework derive
a partial encapsulation of each phase that can be optimize in such a way?


--------

The framework assumes we know the shape of our input tensor and our output tensor. 
We know that a dimension can be stored logically (index * stride + offset) or hardware wise/flattened (/ and %).
Does that mean given the type of data, size of data (bytes) and the length of data, can we derive the optimal way to 
store these values in a tensor, and can we derive a combination tensor (uses both Index * stride + offset and % and /)


------

The dual purpose of the Geometric + FSM + Two Tree structure.

The Geometric + FSM allows us to derive the geometric values given input output tensor, the FSM layer allows us to derive our required
phases of Load Compute and Store, as well as allowing tracking for state variables (like in softmax). The Two Tree is an index derivation layer, this layer
can be abstracted to cuTe/CUTLASS or Triton to professionals whom have experience, but for those whom are learning CUDA like myself, the two tree
index derivation layer serves as a way to understand how the Geometry and FSM structure can map to actual pure CUDA code. The intention is to 
teach the structure of kernels, rather than specific implemenations.

-----

Geometry is abstract enough to apply to NVIDIA CUDA, AMD ROCm and Google's TPU. The FSM is a little more specific but can be abstracted with the right language use.
The reason is possible is because the Geometry and mostly the FSM are based on the problem, not any specific hardware/software.

---------
fundamentally the way data is stored is an inverse of the execution hierarchy what if they were parallel where 

threadIdx.x always correlated to the x dimension in memory space
we can see the register to shared to global not just as a shadow but also as a binary tree, register is the singular parent node and the leads are global
would storing data as binary trees be more useful especially when needing to traverse thru it like what we do with the execution hierarchy?
reminds me of the piano where all threads in the first dimension are mapped to the 3rd
-----------
given some code can we use the table in the reverse way to imply about to geometry?

----

The Geometry and FSM layer must be formed and worded to capture its hardware agnostic perspective.


----

A * B is a dimension + C implies another dimension

blockIdx.y * TILE_SIZE + threadIdx.x * WORK_PER_THREAD + regRow + 1

--- 

At a fundamental level the Two Tree layer of our framework assumes index * stride + offset as the foundational truth. We know that this isnt true, but it instead is a pattern in a much bigger box we havent mapped.
This means the layer needs to step back in a way to understand the whole picture rather than a specific implementation of this picture. We know this is likely the case because we have / and % which belong in the same dimension.
But as it stands now the TTF treats / and % as a special case to the index * stride + offset pattern even though they are a pattern at the same level.




------

the geometry stage goals to capture data processing or computation but what if this geometry space doesn’t just represent  data movement but also data processing where this entire time the geometry was most just overlapped by our data tensor but a processing SOMETHING SHAPE but they are two dimensions that OVERLAP but i’ve assumed we can only see the data tensor but what if there’s an abstract data processing tensor that has been within our dimension view the entire time but we just failed to include it and it’s like gravity where we can’t see it but it effects the around it. this means for every data processing we do, the implicit geometric data processing dimension invisibly overlaps our data holding dimension and transforms it
are the fundamentals of the geometric principles, adding, reducing and DOT product similarity based on

+ is a scalar 
is a 2D operation
/ is within a different dimension hmmm

what i’m getting at is say we have x data point when we add to 3, we are adding three to it in the add minus representation space/dimension and if we say multiply it, in our latent representation space their is also a multiplication/division space which we move say somewhere else in there that changes our base data but it’s just represented different because we modified its specific latent dimension 


data storing is all of our latent dimensions stored, data processing is us change a value in one of those dimensions

higher the dimension in our all possible representation dimensions, if a small change is made in this higher dimension, if changes the lower dimensions more until it reaches the base number with the biggest change 

scalar changes = small in base dimension 

and / can = little changes in this dimension are big changes in scaler dimension

exponential = same deal 

but then there’s also exp and ln inverse relationship, is this just the + and - opposites of our scalar dimension… maybe these are two different dimensions interacting because of their unique relative position to each other 

every dimension has a ruler of increase this way and decrease that way

during computation what if we maintain the natural shape of our problem rather than storing them in separate shared memory tiles

-----------

 To derive a tensor shape by code 

         scores[batch_idx * max_seq_len + token_idx] = 0.0f;
         ->
         [batch_size,max_seq_len]


         smem_q[h * SMEM_STRIDE + d] = q_base[i];
         ->
         [h,SMEM_STRIDE]


         page_table[batch_idx * max_pages + logical_page];
         ->
         [batch_size,max_pages]

         !!!!


--------

Compare the geometric difference between a naive and a optimized kernel. If there is a geometric difference, then the optimization can be expresesd by our geometric dimension.
If there is NOT a difference, that means we are fundamentally missing a dimension that expresses data processing.

The 1D geometry dimension captures data movement but does not capture the whole picture.



--------

The framework cannot create code based on hardware as we have not implemented it so any architecture optimization is not necessarily possible at the moment.
The thing is hardware are just constant across hardware iterations. For the next foreseeable year or two we can say that the DRAM will always be the DRAM, the shared memory
will be shared memory and registers will be registers, the only thing that changes are the numbers. Although there are architecture specific optimizations such as the recent(ish)
tensor cores, but since the framework is based on the geometry of the PROBLEM SPACE which inherently does not change, adding these should just be a technique of sort within the
Finite State Machine layer, and or the Two Tree/Binary Tree layer.

---------

A framework whom does not require a special rule upon a new concept is one that is likely at the right abstraction level (although case by case basis).
If it requires a special rule, the framework is likely, exists within a singular rule, rather than the box the rule is a part of (requires us to step back and understand why).

--------

how it’s stored tells us the index leaf, the type of tensor transformation tells us the equation like is it gonna add or dot which tells us every level info
how it’s stored tells us our execution hierarchy, the tensor operation being done shows our memory hierarchy
together this is what we’re mapping
fundamentally

is parallel programming inherently serialized, yes the computation is done faster because parallel, but we are constrained by our load compute store phases 
if i were to make a gpu i would have the memory storage location each have extra say x bits so that we can do the computation over the memory itself where if we have a 3x3 storage each one has extra information to support holding a computation digit so we can do arithmetic right next to the storage
the execution hierarchy is like shooting three lasers head, when it hits a compute light, it must bend and match the tensor storage layout to properly traverse it

look at the hardware to see what operations are possible
data stored in some layout, and the execution transformed by some operation to match it
a leaf is just a way to transform our execution to the data layout

the whole framework assumes no hardware constraint, hardware is a different dynamic
Image
any kernel is just an operation of functions in a specific order

each of the three (4) axis are a column straight down
tensor operators for loops axis 1
tensor layout for indexing leaf axis 2

Temporal (reuse) lifetime table (optimization) axis 3 
hardware axis 4 
are all optimizations types all specialize from their respective tensor operation axis/light? 

does a combination of say temporal and tensor lights lead to different optimizations

Main two axis

3 tensor operation determine iterator type 

addressing is from data layout 



Two sub axis ? rather its respective optimizations for each respective main axis?

reuse is from derivative of iterator
hardware is derivative of address

Image
add a column for each sub input like say tensor we have the 3 operations and put each respective otpins bottleneck and optimization 


does this for each of the three main dimensions
maybe a cross table between axis memory layout and data addressing (each of their options) mapped out with their flaw and optimization opportunities

is temporal a dot product of the data layout and data address sub-options like arithmetic dot slide
data -> address
operation -> tensor
tensor operation 
vs
data layout 


data layout -> address type (threads)

tensor operation  -> what data we need moved (computation)

. 
TENSOR OPERATION -> LOOP TYPE (AND STATE VARS AND DIM FATE)

DATA LAYOUT -> ADDRESS TYPE (LEAF) 

HARDWARE SPECIFIC OPTIM


we see our thread block grid size 

then our data size and our hardware specific limit for each respective memory level

then our tensor core operations (how the data will be moved)

the TENSOR OPERATIONS are our DATA PROCESSING 
the DATA LAYOUT  in each dimension of our tensors tell us DATA MOVEMENT 

all we need is
our address leaves (data layout)
our iterator type (tensor operations)
our lifecycle tree (temporal reuse)

maybe use 7 optimization equations on each data layout DOT tensor operation to see what is the bottleneck for that specific tile of the table
cause only the data size changes going thru it that structure will always be the same
the 7 equations use hardware numbers ANDDD has both memory and execution operations holy fuck!!!!!

TENSOR OPERATION TELLS US DATA MOVNG OR DATA PROCESSING


ADDRESS TELLS US INDEX


--------

iterators are always thread level, but if we find the blockDim stride, can we make them block level etc? isnt that what we do with grid stride anyway


---------

For the framework, SLIDE, AREA, REDUCE can be misleading to new learners. CONTRACTION, MOVE, PROCESS is better as it follows each tensor operation does, and what 
each iterator actually means so.


>> DATA MOVING (DETERMINED BY TENSOR DATA LAYOUT)
MOVE -> We move data from one memory level to another (MEMORY/MEMORY OPERATION)


>> DATA PROCESSING (DETERMINED BY TENSOR OPERATIONS)
CONTRACTION -> This dimension disappears/contracted (MEMORY/EXECUTION)
AREA -> This dimension survives (MEMORY/EXECUTION)


--------

In a given dimension, can we find the numbers that produced the numbers stored? The numbers themselves may be the products of 
prior arithmetic, so if we find the beginning of this arithmetic pipe, and the end of our arithmetic pipe (our dimension we are going to process),
can we have  acomposite of these functions allowing us to store the composite function 

------------

Binding table is required because tensor operations do not map cleanly to a single loop.

The tensor operation CONTRACTION requires an reduce loop (dimension being reduced) and an AREA loop, for the surviving dimension to accumulate the reduced dimension. These two loops do fundamanetally diferent jobs so they it may be hard to fuse. (Maybe you can though).. a paged loop where the row tells us the reduction part while the area (columns) tell us the AREA part of the loop? Where the columns must be M or N sized (Maybe this requires M and N to be the same dimension) and our row part of our paged loop tells us the reduction output location?


Tiling makes it more complicated where we must SLIDE before we reduce and AREA

Quite literally this is tensor cores. It attempts to fuse the REDUCTION loop and the the AREA loop into one loop, it uses a paged sort of mechanism, and the reason it may be 16 x 16 is because of the idea that I proposed earlier... memory must be ALIGNED arithmetically no fancy addresses like another / %, we need our data to be next to each other (contingious) so that our iterator whom moves 1 at a time... so our loop properly scales to the physical hardware addresses.

This might be able to be side stepped if we understand the stride inbetween each memory location, but that means our data will likely not be stored within the same DRAM sector, which increases instructions required and decreases the value we get by our 32 byte DRAM sector calls if we only use say 16. One way to solve this is we can have a complex memory acess but thenn it has to land in the same DRAM sector, but that would be a sort of rombus shape where we have simple arithmetic at the top, the wide middle is the complexity we have in our address, and the bottom is the same arithmetic simple DRAM, so it makes it rather useless. This must be why an arithmetic address (where the data is stored and can be retrived coalessced/contingous) is required


-------------------


    // Our multi-stride formula is just Index * Stride recursively called. So this formula essentially Index * Stride just presented differently.
    // [Current Batch] * [Num Of Heads in Batch] * [Num of Elements in Head] + [Offset within Head (Batch)] * [Size of each Head] + [Where we are within the head] 
    // It is preferred to be presented this way as it allows us to easliy read our tensor layout of [batch_size, NUM_HEADS, HEAD_DIM]
    // [Z Index * (Size of Y * Size of X) + (Y Index * Size of X) + X Index]



  ------------


  Can the Two Tree Framework (Geometry -> FSM -> Two Tree) provide a shared language for both kernel optimization engineers and compiler engineers. They both approach the same central problem of mappin geometry to code. The kernel optimization engineer focuses on niche cases to push every single ounce of performance, but can this "push to the edge" performance be a........ proper encapsulationn of the abstraction the code is trying to capture? Where every single optimization is just them mapping the geometric + FSM stages + index derivation and how they interact together. For compilers, I assume their job is to match the general case of kernels from geometry -> FSM -> index derviation... This is quite literally like that one paper Low Rank and Sparsity, where low rank is the general pattern (compiler engineers) and the sparsity (edge cases) are the performance optimzation engineers, but at its core, they tackle the same problem, and a unifying language and understanding between them can prove to be beneficial. It's similar to us mapping the memory and execution hierarchy through indexing where we bridge both of these trees to tackle the unified problem both are trying to solve. Where the feedback loop can be a performance engineer finds a niche optmization and is able to use the same language the compiler engineer uses (Geometry -> FSM -> Two Tree) to communicate to the compiler engineer that this edge case may provide a light onto the geometric or even something not mapped yet.

  hm... each edge optmization can can allude to a missed mapping or commonly looked over understanding of our problem space. It is also true that hardware... that hardware plays a major factor where every two years we have new abilities in a way that require us to change our optimzations.... is every hardware benefit at its core a....... mapping of.. mapping of hte problem space but with a better understanidng of how we can use our current architecture to build off of to better map the problem space in its natural dimension? hmmm that might be true but we also know from softmax that geometry is not everything. We had to keep track of not just the geometry of our problem but also running max runningSum etc so there is an algorithmic perspective as well to our Geometry -> FSM -> Two Tree
  perspective. Question is can this algorithmic space be mapped?...

  ... The core reason the framework i believe is possible is because it started with if code can be created, at its fundamental level, it is mapping a higher abstracted pattern that repeats or else we couldnt code it. This means... may allude to the same is for the algorithmic side. I mean we see it every often how a simple algorithmic technique can change the way we structure our kernels. Like for RoPe that I learned from MLA attention, the key insight for RoPE was rather than having fixed values, we instead transfer said values to rotations instead. The key insight being its not the actual values that matter, its the DIFFERENCES we see between them that matters. We also see this in running softmax where it doesnt use rotationn but instead uses A * B + C to rescale its values and it subtracts it by the maximum number to avoid numerical overflow. The important insight being that in softmax, its the relative distance between numbers that matters, n ot the numbers themselves...... There must absolute be a fundamental space to understand what factors need to come into play to know when to apply these techinques. It's the same reason with code exists when a pattern is mapped, the same has to be the same for the algorithmic side where math is a fundamental constant (just like our geometrical problem space) so there must be a way to understand it.

  hmmm but there is also the hardware axis that influences our kernels, but the thing is, thats implementation dependent. Its interesting because the geometric and algoritmic views are hardware agnostic, meaning they are implementation independent and are constant (unless some innovation happens in the field of geometry and algorithms/math, which you never know, it would be very lovely), so its important to understand what is our problem space (Geometry/Algorithmic/Math) vs our implementation space (FSM, Two Tree, CUDA etc). 

  ----------

  RopE, softmax, the new google innovation all are derived that attention requires relative position and not absolute position. Given a space to express values, can we derive the transformations that do not change the dimensions expressions, but allow for optimization.



  ************************************

  No more additions to the parking lot and the framework until the competition is complete as the debt of theory vs application creates a drift that makes it harder to switch from theory to application.
  At this point the framework has the following structure 

  Dimension Fate -> Algorithm Classification -> Access Pattern -> Binding Table -> FSM -> LIfetime Table -> Index Trees -> Code

  Surived/Reduced -> What algorithm for reduced dimension -> How is the tensor stored -> How does dimensions relate to the iterators required from the tensor operation -> Tracking of states and state variables -> Lifetiem Table to indicate re-use -> Two Tree index derivation -> Code

  At the moment our framework represent sm_70. The V5 of the framework supports pipelining (sm_80) but the theory to application drift is signficiant and supercedes the pedalogical implementation of the original Two Tree framework which was its core intention in the first place. 

  Additional theories can be added after 05/01/2026 (competition end)