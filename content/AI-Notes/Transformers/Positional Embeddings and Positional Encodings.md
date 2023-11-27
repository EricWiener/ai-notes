---
tags:
  - flashcards
source: 
summary: 
aliases:
---
Transformers operate on sequences of tokens and they need a way to note the original ordering of the tokens. Positional encodings/embeddings are used to insert information about token position.

The original [[Attention is All You Need]] paper used [[Sinusoidal Positional Encodings]] which used sine and cosine functions to encode positional information. Later approaches mostly use learned embeddings (vs. encodings) and [[Rotary Positional Embedding|RoPE]] is a popular one now.

### Absolute vs. Relative Positional Encodings
[[Sinusoidal Positional Encodings|Absolute positional encoding]]: this is very simple but limits the length of sequences to the length of your positional embeddings. It is also not necessarily meaningful due to the common practice of packing short sentences and phrases together into a single context and breaking up sentences across contexts when forming examples in a batch. **The original transformer paper made use of absolute positional encodings using sine and cosine functions.**

[[Self-Attention with Relative Position Representations|Relative Positional Encodings]]: existing relative positional encodings don't work well with efficient transformers which try to avoid explicitly calculating large matrices (like [[Flash Attention]]).
# FAQ
**What's the difference between a positional embedding and a positional encoding?**
??
A positional embedding is basically a learned positional encoding. 
<!--SR:!2024-02-11,84,290-->

A word embedding is a learned look up map i.e. every word is given a one hot encoding which then functions as an index, and the corresponding to this index is a n dimensional vector where the coefficients are learn when training the model.

A positional embedding is similar to a word embedding. Except it is the position in the sentence is used as the index, rather than the one hot encoding.

A positional encoding is not learned but a chosen mathematical function. $\mathbb{N}\rightarrow\mathbb{R}^n$

**Why do transformers need positional encodings but RNNs don't?**
??
RNNs are unable to factor in long-range dependencies due to their recurrent structure, whereas transformers do not have this problem since they can see the entire sequence as it is being processed. However, this also means that transformers require positional encodings to inform the model about where specific tokens are located in the context of a full sequence. Otherwise, transformer would be entirely invariant to sequential information, considering “John likes cats” and “Cats like John” as identical. Hence, positional encodings are used to signal the absolute position of each token.
[Source](https://jaketae.github.io/study/relative-positional-encoding/)
<!--SR:!2024-02-10,83,290-->