---
tags:
  - flashcards
source: https://arxiv.org/abs/2401.04081
summary: MoE-Mamba reaches the same performance as Mamba in 2.2x less training steps while preserving the inference performance gains of Mamba against the Transformer.
---
See [[AI-Notes/Concepts/Mixture of Experts|MoE]] for an explanation of Mixture of Experts. The model uses x20 the parameters but not all of them are used for inference so it can still run quickly.

[Hacker News Discussion](https://news.ycombinator.com/item?id=38932350)

**Recommendations on how to learn about Mamba:**
I struggled learning about Mamba's architecture but realized it's because I had some gaps in knowledge. In no particular order, they were:
- a refresher on differential equations
- legendre polynomials
- state spaced models; you need to grok the essence of
x' = Ax + Bu
y = Cx
- discretization of S4
- HiPPO matrix
- GPU architecture (SRAM, HBM)

Basically, transformers is an architecture that uses attention. Mamba is the same architecture that replaces attention with S4 - but this S4 is modified to overcome its shortcomings, allowing it to act like a CNN during training and an RNN during inference.

I found this video very helpful: [https://www.youtube.com/watch?v=8Q_tqwpTpVU](https://www.youtube.com/watch?v=8Q_tqwpTpVU)

His other videos are really good too.