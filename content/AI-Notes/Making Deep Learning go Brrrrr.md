---
tags:
  - flashcards
source: https://horace.io/brrr_intro.html
summary: a blog post describing where common areas of bottleneck are when training deep learning models
---
### First principles approach to getting good performance with deep learning
Here are some things the article mentioned:
- If your training loss is lower than your test loss, you're overfitting and there's no point increasing the capacity of your model.
- If your training loss is identical to your validation loss, you shouldn't regularize your model.
# Making your model efficient
Making your deep learning efficient consists of three components:
1. Compute: Time spent on your GPU computing actual floating point operations (FLOPS).
2. Memory: Time spent transferring tensors within a GPU.
3. Overhead: Everything else.

|Performance Regime|Plausible Solutions|
|---|---|
|Overhead-Bound|Tracing, Operator Fusion, don't use Python, JIT tracing|
|Bandwidth-Bound|Operator Fusion|
|Compute-Bound|Use Tensor Cores, give Nvidia more money|

**Why it's important to know what your actual problem is:**
If you are spending your time doing memory transfers, you're in memory-bandwidth bound regime, so there's no point increasing the FLOPs of your GPU.

If you're spending your time doing matmuls (compute-bound regime), then rewriting your model logic into C++ to reduce overhead doesn't make sense.
# Compute
We'd like to maximize the time in the compute-bound regime. You can reduce the overhead or memory costs, but you (mostly) can't reduce the computation required without changing the actual operations you're performing.

**Factory Metaphor:**
We send instructions to our factory (**overhead**), send it materials (**memory-bandwidth**), all to keep our factory running efficiently (**compute**).
![[making-deep-learning-go-brrrrr-20231014173359988.png|300]]

---
So, if our factory increases efficiency faster than the rate at which we can supply it materials, it becomes harder for our factory to achieve its peak efficiency.
![[making-deep-learning-go-brrrrr-20231014173406446.png|300]]
  
Even though our factory's size (FLOPS) doubled - if our bandwidth can't keep up then our performance isn't also going to double
### FLOPS
Modern deep learning accelerators have hardware specialized for matrix-multiplication like Nvidia's Tensor Cores. On an Nvidia GPU you can get 312 FLOPS if doing matrix multiplication but only 19.5 TFLOPS for other FP32 arithmetic.

However, it isn't an issue that the ops besides matrix multiplication are slower since non-matmul ops only occupy a very small fraction of the total FLOPS.

However, if you look at the FLOP counts and runtimes for BERT, you can see that the non-matmul ops make up 0.2% of the FLOPS and are expected to be 16x slower (312/19.5=16), but in reality the **normalization and pointwise ops actually achieve 250x less FLOPS and 700x less FLOPS than our matmuls respectively**:
![[making-deep-learning-go-brrrrr-20231014174106173.png|400]]

> [!NOTE] Why do non-matmul ops take so much more time than they should?
> Because of memory bandwidth. It takes a lot of time to transfer data from DRAM and into SRAM (where compute happens) so even though the ops aren't that slow, the transfer of memory is.

# Bandwidth
Bandwidth costs are the cost paid to move data from one place to another. Some examples of this are:
- CPU -> GPU: data transfer costs
- One node -> another node: network costs
- CUDA global memory -> CUDA shared memory. This is typically referred to as **bandwidth cost** or **memory bandwidth cost**.

Although our factory is where we do the actual work, it's not suitable as a bulk storage unit. A large part of this is that since we're doing actual work here, all the storage is optimized for being fast to actually use (SRAM), instead of having a lot of it.

So, where do we store the actual results and materials? The typical approach is to have a warehouse, probably somewhere where land is cheap and we have a lot of space (DRAM). Then, we can ship supplies to and from our factories (memory bandwidth).

![[making-deep-learning-go-brrrrr-20231014173415043.png|300]]

Th cost of moving stuff to and from our compute units is what's called the ==memory bandwidth== cost. 
<!--SR:!2023-12-20,53,290-->

> [!NOTE]
> Every single time we perform a GPU kernel, we need to move our data from and back to our GPU's DRAM (i.e. our warehouse).

See [[GPU Memory]] for more details.

### Memory-Bound Operations
When we perform an unary operation like `torch.cos`, we need to ship our data from our storage to the warehouse, then perform a tiny bit of computation for each piece of data, and then ship that storage back. Shipping things around is quite expensive. As a result, nearly all of our time here is spent shipping data around, and not on the actual computation itself.

Since we're spending all of our time on memory-bandwidth, such an operation is called a ==memory-bound== operation, and it means that we're not spending a lot of time on compute.
<!--SR:!2024-03-16,111,290-->

![[making-deep-learning-go-brrrrr-20231014194515311.png|200]] ![[making-deep-learning-go-brrrrr-20231014194520569.png|200]]
On the left is what a sequence of pointwise operators might look like. You are sending the same data back and forth between global memory and the compute units. If you use **operator fusion** you can combine operations into one and avoid the extra memory accesses.

---
As an example, you can perform `x.cos().cos()` as either two separate ops which requires 4 reads/writes:
```python
# read from x in global memory, write to x1
x1 = x.cos()

# read from x1 in global memory, write to x2
x2 = x1.cos()
```

or you can do it with one fused op:
```python
# read from x in global memory, write to x2
x2 = x.cos().cos()
```
---

Any 2 PyTorch operators present an opportunity for fusion, thus saving the memory bandwidth costs of reading/writing out to global memory between them.

> [!NOTE] Why activation functions have nearly the same cost.
> A fused `x.cos().cos()` will take nearly the exact same time as calling `x.cos()` by itself. This is why activation functions are nearly all the same cost, despite `gelu` obviously consisting of many more operations than `relu` .

### Measuring memory-bandwidth costs
One common approach to measuring how compute-bound you are is to measure your achieved FLOPS as a percentage of peak FLOPS. For example, if you're achieving 80% of your peak FLOPS, then you know that you're at least 80% compute bound, which is pretty good! The rest of your time is probably spent doing memory-bandwidth operations.

# Overhead
Overhead is when your code is spending time doing anything that's ==not transferring tensors or computing things==.
<!--SR:!2024-03-04,101,290-->

Some examples are:
- Time spent in the Python interpreter.
- Time spent in the PyTorch framework.
- Time spent launching CUDA kernels (but not executing them).

For example, look at this flamegraph profile of PyTorch performing a single addition. That box right there? That's what's performing the actual computation. Everything else is pure overhead:
![[making-deep-learning-go-brrrrr-20231014195625286.png|400]]

Although that chart might make you think people shouldn't be using PyTorch, most of DL is performing massive operations in parallel. PyTorch executes asynchronously which means while it is running a CUDA kernel, it can continue and queue up more CUDA kernels behind it. As long as PyTorch can "run ahead" of the CUDA kernels, most of the framework overhead gets hidden.

### How to tell if you're overhead-bound
There are three methods mentioned:
- Scale your data size and see if your runtime scales proportionally.
- Use the PyTorch Profiler.
- Look at "GPU-Util" in nvidia-smi

**Scale your data size and see if your runtime scales proportionally.** 
Overhead doesn't scale well with data size (while compute and memory do), so the easiest way to tell is to increase your data size and if your runtime doesn't increase proportionally, you're overhead bound.

For example, if you double your batch size but your runtime only increases by 10%, you're likely overhead bound. If you weren't you would expect double the runtime since you need to do double the computation.

> [!NOTE] Caveat
> This isn't strictly the only reason why increasing batch size might not increase computational time accordingly - in certain regimes it also increases computational intensity. For example, in a MLP you're typically doing `[B, D] x [D, D]` matmuls. If B is less than D (say, your batch size is 1 while you hidden dim is 128), then you might negligibly increase your total memory bandwidth, while doubling your compute.

**Use the PyTorch Profiler**
You can use the PyTorch Profiler to see if your CPU kernels are running ahead or behind your GPU kernels.

![[making-deep-learning-go-brrrrr-20231014205137608.png]]
In the above image, the pink lines connect the CPU kernels (top) with the GPU kernels (bottom). There are a lot of gaps between GPU kernels, so this means the GPU is waiting on the CPU overhead.

![[making-deep-learning-go-brrrrr-20231014205221835.png]]
In this example, the CPU is running way ahead of the GPU so you aren't overhead bound.

**Use `nvidia-smi`**
The "GPU-Util" field in nvidia-smi (not "Volatile GPU-Util") basically measures what percentage of the bottom row (referencing the PyTorch Profiler screenshots above) is actually running a GPU kernel.

### Why is there so much overhead in PyTorch?
This is because frameworks like PyTorch allow lots of flexibility to the user and need to decide at runtime "what to do". For example, when you do `a + b`, the following steps need to be taken:

1. Python needs to look up what __add__ dispatches to on `a`.
2. PyTorch needs to determine many attributes of the tensor (such as dtype, device, and whether autograd is needed) to determine which kernel to call.
3. PyTorch needs to actually launch the kernel.

If you don't need this flexibility, you can trace your graph with something like `jit.trace`. You could also use something like TorchDynamo which gives you flexibility and speed.