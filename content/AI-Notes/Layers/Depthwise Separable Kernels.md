---
tags: [flashcards]
aliases: [Depthwise Convolution]
source: https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728
summary: reduces computation by using a depthwise layer for filtering and a separate pointwise layer (1x1 kernel) for creating new features.
---
### Overview

![[depthwise-pointwise-convolution.png]]
Example picture of the overall process of depthwise → pointwise convolution 

> [!NOTE]
> A depthwise separable convolution decomposes a standard convolution into a **depthwise convolution** (applying a single filter for each input channel) and a **pointwise convolution** (combining the outputs from depthwise convolution across channels)

### Traditional Convolution
![[traditional-convolution.png]]
- In the above example we have a 12x12 image with 3 channels (RGB). We then apply a 5x5 kernel in order to get an 8x8 output. The kernel also needs to have 3 channels (one for each layer of the input). The number of kernels we apply decides the number of output channels.
- Each time the kernel is applied to a section of the image, it will operate on a 5x5 region of the image that has 5x5x3 values. It will then compute an inner product of the 5x5x3 values of the image with the 5x5x3 weights in the kernel to produce a single scalar in the output.

### [[Depthwise Convolution]]
![[depthwise-convolution.png]]
- In a depthwise convolution, you apply filters without changing the number of channels.
- Each 5x5x1 kernel iterates 1 channel of the image (**note: 1 channel,** not all channels). You then stack the results from each of the kernels to get an 8x8x3 result (which has the same number of channels as the input).
- Instead of each kernel computing a 5x5x3 multiplications/adds each time it is applied, the kernels perform 5x5x1 multiplications/adds.

### [[Pointwise Convolution]]
![[pointwise-convolution.png]]
- The original convolution transformed a 12x12x3 image into a 8x8x1 image. However, the depthwise convolution currently transforms the 12x12x3 image into a 8x8x3 image.
- We can then apply a 1x1 kernel that iterates through every point in the output from the depthwise convolution. In this example, it will have 3 channels. This will produce an 8x8x1 final output from the 8x8x3 intermediate result.

### What’s the point of creating a depthwise separable convolution?
Example: taking a 12x12x3 image and getting an 8x8x256 result.
- In the original convolution we had 256 5x5x3 kernels that produce 8x8 values. That is a total of 256x3x5x5x8x8=1,228,800 multiplications.
- In the separable convolution, the depthwise convolution has 3 5x5x1 kernels that produce 8x8 values (3x5x5x8x8 = 4,800 multiplications). The pointwise convolution has 256 1x1x3 kernels that produce 8x8 values. That’s 256x1x1x3x8x8 = 49,152 multiplications. You then add up these two values and that’s a total of 53,952 multiplications.
- The separable convolution does a lot less computation.

The main difference is this: in the normal convolution, we perform a more complex transformation every time (more parameters involved). And every transformation uses up 5x5x3x8x8=4800 multiplications. In the separable convolution, we perform a simpler 5x5 transform in the depthwise convolution. Then, we take the transformed image and use the 1x1 kernels (with less learnable parameters) to change the number of channels.

Using a depthwise separable convolution can improve speed, but also results in having fewer learnable parameters and learning less complex functions.

Note that in most ML frameworks you can set the “depth multiplier” argument to increase the number of output channels in the depthwise convolution to increase the number of parameters learned.

> [!NOTE] Unlike spatially separable kernels, depthwise separable kernels work for all kernels (not just special cases).
> An example of a spatially seperable kernel (ex. splitting a $3 \times 3$ Gaussian Kernel into two $1 \times 3$ filters convolved with each other). However, you are not always guaranteed that multiplying two smaller matrices will produce a larger matrix, so only some kernels are spatially seperable.


# MobileNet Explanation
![[MobileNet#Depthwise Convolutions]]

# Caveots
- Depthwise seperable convolutions are not compatible with Application-Specific Integrated Circuit (ASIC). These are chip customized for a particular use, rather than intended for general-purpose use.
