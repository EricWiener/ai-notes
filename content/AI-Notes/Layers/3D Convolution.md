---
tags: [flashcards]
source: https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215
summary: 3D convolutions are a generalization of 2D convolutions where the number of kernels in a filter is less than the number of channels in the input. This means the filter moves in three dimensions.
---

### Layer (Filter) vs. Channels (Kernels)
![[layer-vs-kernel.png|300]]
A “Kernel” refers to a 2D array of weights. The term “filter” is for 3D structures of multiple kernels stacked together. For a 2D filter, filter is same as kernel. But for a 3D filter and most convolutions in deep learning, a filter is a collection of  ==kernels==. Each kernel is unique, emphasizing different aspects of the input channel.
<!--SR:!2024-03-08,271,310-->

### 2D Convolution (Refresher)
Another way to think about 2D convolution: thinking of the process as sliding a 3D filter matrix through the input layer. Notice that **the input layer and the filter have the same depth (channel number = kernel number)**. The 3D filter **moves only in 2-direction**, height & width of the image (That’s why such operation is called as 2D convolution although a 3D filter is used to process 3D volumetric data). **The output is a one-layer matrix**.
![[3d-convolution-20230322115927285.png|300]]

Let’s say the input layer has _Din_ channels, and we want the output layer has _Dout_ channels. What we need to do is to just apply _Dout_ filters to the input layer. Each filter has _Din_ kernels. Each filter provides one output channel. After applying _Dout_ filters, we have _Dout_ channels, which can then be stacked together to form the output layer.
![[2d-convolution-diagram.png|600]]


> [!NOTE] 2D Convolution
> It’s a 2D convolution on a 3D volumetric data. The filter depth is same as the input layer depth. The 3D filter moves only in 2-direction (height & width of the image). The output of such operation is a 2D image (with 1 channel only).

### 3D Convolution
They are the generalization of the 2D convolution. Here in 3D convolution, the filter depth is smaller than the input layer depth (kernel size < channel size). As a result, the 3D filter can move in all 3-direction (height, width, channel of the image). At each position, the element-wise multiplication and addition provide one number. Since the filter slides through a 3D space, the output numbers are arranged in a 3D space as well. The output is then a 3D data.
![[3d-convolution-20230322120231699.png|400]]

