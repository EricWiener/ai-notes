---
tags:
  - flashcards
source: https://arxiv.org/abs/1703.06211
summary: 
publish: true
---

[YouTube Video](https://www.youtube.com/watch?v=6TtOuVJ9GBQ)

**Deformable convolutions** add 2D offsets to the regular grid sampling locations in the standard convolution. It enables free form deformation of the sampling grid. The offsets are learned from the preceding feature maps, via additional convolutional layers. Thus, the deformation is conditioned on the input features in a local, dense, and adaptive manner.

![[deformable-convolution-20221231124345097.png]]

Deformable convolutions are a generalization of [[Dilated Convolution]]s where you learn the offset instead of having a fixed offset. [Source](https://youtu.be/LMZI8DDyltQ?t=2672).