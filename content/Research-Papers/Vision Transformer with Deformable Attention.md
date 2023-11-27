---
tags: [flashcards]
aliases: [DAT]
source: https://arxiv.org/abs/2201.00520
summary: present Deformable Attention Transformer, a general backbone model with deformable attention for both image classification and dense prediction tasks.
---

This paper improves the [[Deformable Attention]] introduced in [[Deformable DETR]] and makes it better suited to use throughout the entire transformer.

# What's the difference between DAT and [[Deformable DETR]]?

**DAT is used in the vision backbone while D-DETR is used in the detection head**
DAT's deformable attention serves as a feature extractor in the vision backbones while the one in Deformable DETR which replaces the vanilla attention in DETR with a linear deformable attention, plays the role of the detection head.

**D-DETR is closer to convolution than attention**

**D-DETR is slower and consumes more memory**

**[[Deformable DETR]]'s code has a custom C++ op while DAT doesn't**
> This is because the spatial sampling operation in DAT is relatively simple and can be directly implemented by `F.gridsample(feature, pos)` with a feasible speed, while Deformable-DETR provides a more optimized CUDA OP version for different numbers of keys. Therefore, there could be an optimized CUDA implementation for DAT if low latency is in demand. [Source](https://github.com/LeapLabTHU/DAT/issues/18#issuecomment-1264688573)

