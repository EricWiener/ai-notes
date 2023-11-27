---
tags: [flashcards, eecs498-dl4cv]
source:
summary:
aliases: [upconvolution, deconvolution, fractionally strided convolution, backward strided convolution]
---

[[Upsampling]] and [[Max Unpooling]] don't have learnable parameters. Transposed convolution (also called "upconvolution" or "deconvolution") does have ==learnable parameters==. Additionally, transpose convolution is able to change the number of ==channel dimensions== (unlike the other methods).
<!--SR:!2027-04-14,1291,338!2027-04-15,1292,338-->

When performing a convolution, if we use a stride == 1 (and same padding), the output has the same size. If we use a stride > 1, we will downsample the image ("learnable downsampling" since we learn the weights). Now, we want to somehow use a stride < 1 for "learnable upsampling."

In a transposed convolution, you have a filter that you ==scale== by values in the image (scalar multiplication). You then copy the scaled filter to the output and sum up anywhere where you have overlapping values.
<!--SR:!2024-05-14,597,330-->

![[transposed-convolution-1d.png]]
Transposed Convolution: 1D Example

![[transposed-convolution-2d.png]]
Transposed convolution 2D example

In the 2D example above, the bottom-left component is generated by multiplying the filter (this filter is learned by the model) by the bottom-left scalar in the image. You then copy over these values to the output. You repeat this for the other elements in the input and then sum up anywhere there is overlap.

One issue with this is you sometimes get a "checkerboard" artifacts.

### Why is this called transposed convolution?
![[transposed-convolution-math.mp4]]

![[transposed-conv-math.png]]
Transposed convolution is called this because regular convolution can be represented as matrix multiplication with a filter $X$ and input $\vec{a}$. Transposed convolution can be represented as $X^T\vec{a}$.