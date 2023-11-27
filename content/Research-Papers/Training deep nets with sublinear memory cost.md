---
tags: [flashcards]
source: https://ar5iv.labs.arxiv.org/html/1604.06174
summary: recompute forward passes to avoid needing to store each intermediate feature map
---

We propose a systematic approach to reduce the memory consumption of deep neural network training. Specifically, we design an algorithm that costs $O(\sqrt{n})$ memory to train a $n$ layer network, with only the computational cost of an extra forward pass per mini-batch.

### Method
During the backpropagation phase, we can re-compute the dropped intermediate results by running forward from the closest recorded results. To present the idea more clearly, we show a simplified algorithm for a linear chain feed-forward neural network in Alg. [1](https://ar5iv.labs.arxiv.org/html/1604.06174#algorithm1 "Algorithm 1 ‣ 4.1 General Methodology ‣ 4 Trade Computation for Memory ‣ Training Deep Nets with Sublinear Memory Cost"). Specifically, the neural network is divided into several segments. The algorithm only remembers the output of each segment and drops all the intermediate results within each segment. The dropped results are recomputed at the segment level during back-propagation. As a result, we only need to pay the memory cost to store the outputs of each segment plus the maximum memory cost to do backpropagation on each segment.