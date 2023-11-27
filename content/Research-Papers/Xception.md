---
tags: [flashcards]
source: https://ar5iv.labs.arxiv.org/html/1610.02357
aliases: [Deep Learning with Depthwise Separable Convolutions]
summary: replaces Inception modules in [[Going Deeper with Convolutions|Inception Networks]] with [[Depthwise Separable Kernels]].
---

![[xception-network-architecture.png]]
The Xception architecture: the data first goes through the entry flow, then through the middle flow which is repeated eight times, and finally through the exit flow. Note that all Convolution and SeparableConvolution layers are followed by batch normalization (not included in the diagram). All SeparableConvolution layers use a depth multiplier of 1 (no depth expansion).
