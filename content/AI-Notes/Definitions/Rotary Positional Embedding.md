---
tags:
  - flashcards
source: https://youtu.be/GQPOtyITy54
summary: a type of relative positional embedding that works for transformers that don't explicitly calculate the $N \times N$ attention matrix
aliases:
  - Rotary Embeddings
  - RoPE
---
Rotary Positional Embedding (RoPE) is a new type of position encoding that unifies absolute and relative approaches. It either matches or surpasses all other methods currently available for injecting positional information into transformers.

RoPE embeddings will rotate vectors at varying amounts depending on their position in the sequence and the channel dimension (it breaks up the token embeddings into $D/2$ groups of channel = 2 and then rotates each of the $D/2$ groups using a 2D rotation matrix that is different for each dimensional group).

> [!NOTE]
> Rotary Positional Embeddings is designed to be an easy to implement, and generally-applicable method for relative position encoding that works for both vanilla and “efficient” attention.

**What do RoPE Embeddings do?**
??
RoPE embeddings will rotate keys and query vectors depending on their position in the sequence and the channel dimension. This improves model generalization for sequences longer than it is trained on.
<!--SR:!2023-12-11,11,250-->

### Motivation
When calculating the attention matrix, you want it to have two characteristics:
- Tokens that have similar token embeddings should have a higher score.
- The score should be higher for words that are close together. The further words are from each other, the less likely they are related.
<!--SR:!2023-11-01,4,270-->

Each entry in the attention matrix is the dot product of some query and some key vector.

Below shows the transformer self-attention operation computing the attention scores (pre-softmax) between 2D vectors for queries (orange) and keys (blue). The blue and green grids on the right show **what we are trying to achieve with RoPE embeddings.**
![[RoPE (Rotary positional embeddings) explained_ The positional workhorse of modern LLMs 2-40 screenshot.png]]

The top-right visualization shows how the attention matrix ideally captures the token embedding similarity between tokens $(m, n)$. This is calculated via the similarity between the radial components of tokens $m$ and $n$  ($||q_m|| \cdot ||k_n||$). The token similarity is independent of the angle of the individual tokens.

The visualization below shows how attention matrix captures the positional similarity of tokens when RoPE embeddings are used through their relative angle. The angle component only depends on the position in the sequence and is independent of the actual token.
![[RoPE (Rotary positional embeddings) explained_ The positional workhorse of modern LLMs 3-45 screenshot.png]]
[Source](https://youtu.be/GQPOtyITy54?t=225)
### Issues with sinusoidal encodings
![[RoPE (Rotary positional embeddings) explained- The positional workhorse of modern LLMs-00.04.53.129-00.06.21.131-sinusodial-encodings-issue.mp4]]
# Implementation
RoPE will rotate vectors at a different frequency depending on their position in the sequence and channel dimension.
### Using rotations for adding positional information
![[RoPE (Rotary positional embeddings) explained- The positional workhorse of modern LLMs-00.08.53.138-00.11.33.417-rope-overview.mp4]]
### Explanation behind RoPE equation
$$a_{m, n}=q_m^T k_n=x_m^T W_q^T\left[R_{\Theta, d}^m{ }^T R_{\Theta, d}^n\right] W_k x_n$$

![[RoPE (Rotary positional embeddings) explained- The positional workhorse of modern LLMs-00.12.36.184-00.13.04.770-math-behind-rope-equation_2.mp4]]

