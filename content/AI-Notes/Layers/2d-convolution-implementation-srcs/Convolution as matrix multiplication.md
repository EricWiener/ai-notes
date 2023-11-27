---
tags: [flashcards]
aliases: [im2col]
source: https://rancheng.github.io/Sparse-Convolution-Explained/
summary: this is how deep learning frameworks compute convolutions as matrix multiplication
---

[Another good source](https://towardsdatascience.com/how-are-convolutions-actually-performed-under-the-hood-226523ce7fbf)

The following diagram shows how convolution can be computed using matrix multiplication. You unroll the kernel into a sparse matrix and then apply it to the flattened input. This results in the same outputs (just different shapes).

In the original convolution you do ($w_{0, 0} \cdot x_0 + w_{0, 1} \cdot x_1 + ...$). In the matrix multiplication version you multiply the row of the sparse matrix C with the column of the input. This will also perform ($w_{0, 0} \cdot x_0 + w_{0, 1} \cdot x_1 + \ldots$).  The same computation is done but in a different way.
![[image-20230711113401794.png]]
> This is a typical step what people do when we are performing a convolution operation in computer, the weight matrix is unrolled based on the size of kernel and the size of the input size with zero padding, for a `3x3` size kernel, the first second and third row is shift repeatedly `w` times where `w` is the size of the image width. This is equivalent to do the sliding window in normal convolution.

![[im2col.gif]]
[Source](https://medium.com/geekculture/3d-sparse-sabmanifold-convolutions-eaa427b3a196)
