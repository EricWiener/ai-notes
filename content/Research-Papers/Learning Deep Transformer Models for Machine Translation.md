---
tags:
  - flashcards
summary: suggest moving from using post-norm to pre-norm for LayerNorm
source: "[[Learning Deep Transformer Models for Machine Translation, Qiang Wang et al., 2019.pdf]]"
aliases:
  - Pre-Norm
  - pre-norm
---

![[pre-vs-post-norm.png|500]]
In the above figure you can see [[Layernorm]] (LN) is after the residual connection for post-norm but before it for pre-norm.
### Pre-norm over post-norm
Both methods work okay for not-so-deep networks, but the paper says that pre-norm is better for deep residual networks. Prenorm is more efficient for training than post-norm if the model goes deeper since on the backward pass the gradient can flow directly through the residual connection instead of having to go through the LayerNorm.