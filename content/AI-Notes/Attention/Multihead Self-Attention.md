---
tags: [flashcards]
source:
summary:
---

You can split the input $X$ into ==different chunks==. You can then pass each chunk through a separate self-attention layer. You then concatenate the output results.
<!--SR:!2028-06-20,1801,359-->

Multihead Attention will split the input along the channel dimension (you still pass the same number of tokens to each head). Ex: if the dimension of your input $X$ is `embed_dim`, then `embed_dim` will be split across `num_heads` (i.e. each head will have dimension `embed_dim // num_heads`). 
![[multi-head-self-attention.png]]

**Hyperparameters:**
- Use $H$ independent "Attention Heads" in parallel.
- Need to choose the dimension of the query vector $D_Q$ (these are the internal key vector dimensions). The key matrix is of size $D_X \times D_Q$. $D_X$ is determined by the input, but $D_Q$ is specified. We encode our input-vectors using either a one-hot vector, word embedding, or another method. $D_Q$ specifies the dimensions of our embedding.
- This is similar to the idea of [[Grouped Convolutions]] that we saw in [[ResNeXt]].

**Benefit of multiple heads**:
- Each of the heads starts with different randomly initialized weights, so having multiple heads will allow you to learn different representations.
- [[Attention is All You Need]] says: "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this."
- This means that when you have a single head and compute the weighted average (via a weighted combination of the value vectors according to the alignment scores), you will lose representational ability.
- [SO Answer](https://datascience.stackexchange.com/a/55722/70970): For a given layer, with only one attention head, the weighted averaging performed by the mechanism prevents you from being able access (differently transformed) information from multiple areas of the input within that single layer. In general, using multi-head is a common way to increase representational power in attention-based models, for instance see _Graph Attention Networks_ by Velickovic et al. The basic idea is that, although attention allows you to "focus" on more relevant information and ignore useless noise, it can also eliminate _useful_ information since usually the amount of information that can make it through the attention mechanism is quite limited. Using multiple heads gives you space to let more through.
- "Multi-head attention can be seen as an ensemble of low-rank vanilla attention layers." [CMU Blog](https://blog.ml.cmu.edu/2020/03/20/are-sixteen-heads-really-better-than-one/).
- Different heads learn different things [[Analyzing Multi-Head Self-Attention_ Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned, Elena Voita et al., 2019.pdf]]
    - Ex: "We observe the following types of role: positional (heads attending to an adjacent token), syntactic (heads attending to tokens in a specific syntactic dependency relation) and attention to rare words (heads pointing to the least frequent tokens in the sentence)."

### Math
$$\begin{aligned}
& \operatorname{MultiHead}(Q, K, V)=\operatorname{Concat}\left(\text { head }_1, \ldots, \text { head }_{\mathrm{h}}\right) W^O \\
& \text { where head }_{\mathrm{i}}=\operatorname{Attention}\left(Q W_i^Q, K W_i^K, V W_i^V\right)
\end{aligned}$$
where:
- $W_i^Q \in \mathbb{R}^{d_{\text {model }} \times d_k}$: applied to embed the query.
- $W_i^K \in \mathbb{R}^{d_{\text {model }} \times d_k}$: applied to embed the key.
- $W_i^V \in \mathbb{R}^{d_{\text {model }} \times d_v}$: applied to embed the value.
- $W^O \in \mathbb{R}^{h d_v \times d_{\text {model }}}$: this is an output projection matrix that is applied to the concatenated output of the different heads and then produces the final output (it is a linear layer).