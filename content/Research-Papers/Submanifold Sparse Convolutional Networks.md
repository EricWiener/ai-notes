---
tags: [flashcards]
aliases: [Sparse Manifold Convolutions, Sparse Convolutions, Submanifold Sparse Convolution]
source: https://arxiv.org/abs/1706.01307
summary: this paper introduced submanifold convolutions which operate on sparse data and don't dilate the sparsity with subsequent layers.
---
[Useful blog post](https://medium.com/geekculture/3d-sparse-sabmanifold-convolutions-eaa427b3a196)

This paper introduces a new way to perform convolutions on sparse data (a lot of your input is empty space). Existing approaches for handling sparse data "dilate" the sparse data in every layer because they implement a "full" convolution. This paper introduces a technique to keep the same sparsity pattern throughout the layers of the network without dilating the feature maps. The introduce two new convolution operators: sparse convolution and valid sparse convolution (aka submanifold convolution). In the experiments they performed on 2D digit recognition and 3D shape recognition, networks using SC and VSC achieve SOTA with computation and memory usage reduced ~50%.

**Submanifold definition:** Submanifolds refer to sparse input data, such as one-dimensional curves in two-dimensional space, two-dimensional curved surfaces, and point clouds in three-dimensional space. They do not occupy the entire space in which they are located.

The paper refers to their networks as submanifold convolutional networks, because they are optimized to process low-dimensional data living in a space of higher dimensionality.

### Why submanifold sparse convolutions when we already have 3D convolutions?
[[3D Convolution]] makes use of 3-dimensional filters. Pictures and videos contain dense data and it makes sense to apply convolutions directly on this data. However, when the data is sparse (like in a point cloud with lots of empty space), directly applying traditional convolutions will result in a lot of wasted computation.

As you apply more convolutions, the sparsity of the data cannot be maintained (this is referred to as the "**submanifold expansion problem**"). If your input data has a single active site (non-zero location), then after applying a 3x3 convolution, there will be 3x3 active sites since there will be 3x3=9 times when your convolution overlaps with the single active site. In the diagram below you can see in your input (left) with a single active site (green), when the 3x3 kernel (blue) is applied to the image, when the center of the kernel is at locations 1-9, the kernel will overlap with the active site somewhere. Therefore, this will produce a non-zero value for all of those locations in the output.
![[kernel-applied-to-single-active-site.png|800]]

Here's another example from the paper, 
![[screenshot 2023-07-11_12_01_25@2x.png]]
> Figure 1: Example of “submanifold” dilation. Left: Original curve. Middle: Result of applying a regular 3×3 convolution with weights 1/9. Right: Result of applying the same convolution again. The example shows that regular convolutions substantially reduce the sparsity of the feature maps.

### Types of submanifold sparse convolutions
There are two parts of the convolutions: **sparse convolutions** and **submanifold convolutions**. The difference between both operations is in how they handle active sites: sparse convolutions will apply a kernel to the input at all locations where any part of the kernel overlaps an active site. Submanifold convolutions, however, will only be applied at locations where the center of the kernel overlaps with an active site.

### Sparse convolutions
These are like normal convolutions except if the convolution doesn't overlap with any active sites, it assumes the input from those sites is exactly zero. It is often used mainly in downsampling layers since it has a higher computational cost but is able to spread feature information.

**Benefits**:
They dilate all sparse feature to their $\text{kernel-size}^3$ neighbors for 3D conv and $\text{kernel-size}^2$ for 2D conv. This gives them better receptive fields and information flow.

**Cons**:
They have a higher computation cost because the sparsity gets dilated so you end up performing more and more computation at later layers. This results in slower inference speed and a larger GPU memory cost.

### Submanifold convolutions
These are a restricted form of sparse convolutions where an output site is active if and only if the site at the corresponding site in the input is active (i.e., if the central site in the receptive field is active). When an output site is determined to be active, its output feature vector is calculated the same way as for sparse convolutions.

**Benefits**:
You input sparse features and you get sparse outputs. It is also very efficient.

**Downsides of submanifold sparse convolutions**: ==poor representative ability==.
If we restrict the output of the convolution only to the set of active input points, it is hard to propagate global information since convolution only takes place at valid voxels. Features cannot reach the voxels that have big gaps in between (as shown below). This results in disconnected information flow and very limited receptive fields.
![[screenshot 2023-07-11_15_18_20@2x.png|300]]
<!--SR:!2023-12-15,48,230-->

### Implementation
Traditional 2D convolutions are implemented as matrix multiplication under the hood using [[Convolution as matrix multiplication|im2col]] which is easier to parallelize and compute efficiently. This same approach doesn't work on sparse data. Instead, sparse convolution will collect all atomic operations w.r.t convolution kernel elements and save them in a Rulebook as instructions of computations which are then executed in parallel to get the speed-up. 
