---
tags:
  - flashcards
source: https://arxiv.org/abs/2205.14135
summary: FlashAttention lets you train networks faster and with a longer context length.
aliases:
  - Fast and Memory-Efficient Exact Attention with IO-Awareness
---

> [!NOTE] TLDR
> FlashAttention: fast and memory-efficient algorithm for exact attention. It uses tiling (==break apart softmax==) and recomputation (==recomputes attention during backward pass==). It enables you to train models faster with longer sequences.
<!--SR:!2023-12-17,50,290!2024-03-15,110,290-->

[Stanford MLSys YouTube Talk](https://youtu.be/gMOAud7hZg4)

Three main reasons to model longer sequences:
- NLP: A longer context length is required to understand books, plays, and instruction manuals.
- Computer vision: higher resolution can lead to better, more robust insights into the image context.
- New areas: if you can work with longer sequences, you can work with time series, audio, video, medical imaging data, etc.
[Source](https://ai.stanford.edu/blog/longer-sequences-next-leap-ai/#:~:text=It's%20possible%20that%20longer%20sequences,on%20so%20much%20more%20information!)

# Attention Computation
Attention doesn't scale well with respect to sequence length (quadratic computational complexity because it captures dependencies between all pairs of input and output). Training slows down with a longer context length and you can run out of memory.

$\mathbf{O}=\operatorname{Dropout}\left(\operatorname{Softmax}\left(\operatorname{Mask}\left(\mathbf{Q K}^{\mathbf{T}}\right)\right)\right) \mathbf{V}$ 
![[flash-attention-20231013091005758.png]]
The values $Q, K, V, O$ are all $N \times d$ where $N$ is the sequence length (ex. 2k) and $d$ is the dimension of the attention (usually 64 or 128). These dimensions are not so large but the intermediate matrices are $N \times N$ which is very large for long sequence lengths. This requires lots of reads/writes from slow GPU HBM (high bandwidth memory).

> [!NOTE]
> FlashAttention uses tiling and recomputation to reduce GPU memory IOs (reads/writes). It is fast (3x) ad memory efficient (10-20x) and computes exact attention (no approximation). This allows training with longer sequences and getting a higher quality model.

# Related Work
**Data Movement is All You Need: A Case Study on Optimizing Transformers**
This paper analyses transformers and found that a lot of the computation is spent on moving to and from GPU memory. Not a lot of time on computation - a lot of the time is spent on memory IO.

![[screenshot 2023-10-14_09_12_48@2x.png]]

A lot of the time in a naive implementation of attention is spent on the Softmax and Dropout (low computation) and not on the matrix multiplies (high computation).

**Sparse Attention**
You don't attend to all regions and just attend to some points.

**Low Rank Approximation**
You factor the attention matrices into two low rank parts and then use these.

> [!NOTE]
> Both sparse attention and low rank approximation have not been widely adopted because you sacrifice model performance (can't compute exact attention) and the training isn't that much faster.

# GPU Compute Model + Memory Hierarchy
Diagram below is for a machine with N GPUs (ex. machine with 8 A100s).
![[flash-attention-20231013093616741.png]]
Memory Hierarchy:
- Main Memory (CPU DRAM): this is CPU memory (ex. 1 TB).
- GPU HBM: this is what you thing of as main memory (ex. 40GB on A100).
- GPU SRAM: much smaller and much faster. This does matrix multiply, softmax, etc. (ex. 20 MB)

You move data back and forth between SRAM and HBM. SRAM is much smaller than HBM but an order of magnitude faster.

> [!NOTE] HBM
> HBM stands for high bandwidth memory and is a type of memory interface used in 3D-stacked DRAM. The paper could have generalized better if they referred to GPU HBM as GPU DRAM.

# Flash Attention
How to reduce HBM reads/writes: compute by blocks (has been around since 1970). This allows you to load a block from HBM to SRAM and keep it there without having to move it back and forth between HBM and SRAM.

Challenges to computing attention by blocks:
- Compute softmax reduction without access to the full input. Softmax needs an entire row to normalize all values.
- Backward pass without the large attention matrix from the forward pass.

Techniques to solve the problem:
- **Tiling**: restructure the algorithm to load block by block fro HBM to SRAM to compute attention.
- **Recomputation**: don't store the attention matrix from the forward pass and instead recompute it in the backward pass.

They use a fused CUDA kernel for fine-grained control of memory accesses.

### Tiling
You can break apart a softmax into multiple parts (similar to what [[LLama v2 Adapter]] does with its zero-init attention with a gating factor). You just need to apply a scaling factor to each of the sub-parts before re-combining them to make sure your final result is normalized correctly:
$$\operatorname{softmax}\left(\left[A_1, A_2\right]\right)=\left[\alpha \operatorname{softmax}\left(A_1\right), \beta \operatorname{softmax}\left(A_2\right)\right]$$
You can then normalize the attention scores -> attention weights and combine the attention weights and values to get the outputs in a block-by-block fashion using:

$$\operatorname{softmax}\left(\left[A_1, A_2\right]\right)\left[\begin{array}{l}
V_1 \\
V_2
\end{array}\right]=\alpha \operatorname{softmax}\left(A_1\right) V_1+\beta \operatorname{softmax}\left(A_2\right) V_2$$

**Details on splitting apart Softmax:**
A numerically stable softmax can be calculated with:
$$m(x):=\max _i \quad x_i, \quad f(x):=\left[\begin{array}{lll}
e^{x_1-m(x)} & \ldots & e^{x_B-m(x)}
\end{array}\right], \quad \ell(x):=\sum_i f(x)_i, \quad \operatorname{softmax}(x):=\frac{f(x)}{\ell(x)}$$
See [[Online normalizer calculation for softmax]] for more details.

Therefore, they keep track of some extra statistics $(m(x),\,\ell(x))$ and do some fancy math and are able to perform softmax block-by-block. See the paper for the algorithm they use.

**Computation Pipeline:**
1. Load inputs by blocks from HBM to SRAM.
2. On chip, compute attention output wrt that block.
3. Update output in HBM by scaling.

### Recomputation (backward pass)
You want to recompute attention with respect to a small block already in SRAM without making any accesses to HBM. They can recompute this quickly by storing the **softmax normalization factors** from the forward pass. This allows doing a per-block computation of attention without needing to reference the other blocks. This uses more FLOPs but speeds up the backward pass.

# Empirical Validation
They want to validate:
- They get faster end-to-end training of transformers
- They get memory savings (want memory to scale linearly instead of quadratically).

![[screenshot 2023-10-14_09_34_50@2x.png]]
They see a speed up over standard PyTorch attention at different sequence lengths, on A100, of 2-4x.

They see a 10-20x memory reduction since they don't need to write down the full $N^2$ matrix.

FlashAttention beats previous MLPerf record for training BERT and is 3x faster than HuggingFace's BERT.

They train GPT-3 with longer sequences and a longer sequence length leads to lower [[Perplexity]] (better). This shows the benefit of scaling up to longer sequences.

# Future Directions
- Optimize IO at inference time.
- Support distributed training.
- Be more hardware-aware about other ops.
- Use longer sequence lengths.

FlashAttention is not very useful for iterative decoding where you decode one token at a time ($1 \times N$ instead of $N \times N$). The bottleneck for iterative decoding is loading the key and value matrices from cache. However, there could be ways to speed this up.

Similar ideas could be applied to alternate forms of attention that don't use Softmax. Softmax is difficult because it couples the entire row, but things like element-wise ReLU would be even easier to implement.