---
tags: [flashcards]
aliases: [CSPNet]
source: [[Wang_CSPNet_A_New_Backbone_That_Can_Enhance_Learning_Capability_of_CVPRW_2020_paper.pdf]]
summary:
---

[YouTube Notes](https://youtu.be/a0sxeZALxzY)

The authors propose Cross Stage Partial Network (CSPNet) to mitigate the problem that computer vision models require heavy inference computations from the network architecture perspective. We attribute the problem to the **duplicate gradient information** within network optimization (ex. in DenseNet multiple blocks will perform convolutions on the same inputs). The proposed networks respect the variability of the gradients by integrating feature maps from the beginning and the end of a network stage (only performing convolutions on some inputs and directly concatenating the outputs with the remaining inputs), which, in our experiments, reduces computations by 20% with equivalent or even superior accuracy on the ImageNet dataset.

### Problem
Cross Stage Partial Networks are trying to solve two problems:
1. There is limited computational resources available on edge devices. Edge devices are devices that collect data (they are part of a network and the edge refers to being on the outside layer of the network vs the central servers).
2. [[Depthwise Separable Kernels]] are not compatible with Application-Specific Integrated Circuit (ASIC) for edge-computing devices. An ASIC is a chip customized for a particular use, rather than intended for general-purpose use. [Wikipedia](https://en.wikipedia.org/wiki/Application-specific_integrated_circuit).
CSPNets are computationally efficient components that enable "ResNet, ResNext, DenseNet" to be deployed on both CPUs and mobile GPUs.

### Typical Networks
In mainstream CNN architectures, the output is usually a linear or non-linear combination of the outputs of intermediate layers.

In general, there are two fundamental ways that one could use skip connections through different non-sequential layers to combine inputs from previous feature maps [Source](https://theaisummer.com/skip-connections/):
a) **addition** as in residual architectures ([[ResNeXt#Skip Connection]]
b) **concatenation** as in densely connected architectures ([[DenseNet]]).

In the following two equations, R and D respectively represent the computation operators of the residual layer and dense layer, and these operators often composed of 2∼3 convolutional layers.

**[[ResNeXt]] architecture equation:**
$$
\begin{aligned}
x_{k} &=R_{k}\left(x_{k-1}\right)+x_{k-1} \\
&=R_{k}\left(x_{k-1}\right)+R_{k-1}\left(x_{k-2}\right)+\ldots+R_{1}\left(x_{0}\right)+x_{0}
\end{aligned}
$$
- The output of a ResNext layer is the output of a residual block added to the skip connection. The skip connections theoretically means you are adding all the previous feature maps and original input to the model.

**[[DenseNet]] architecture equation:**
![[densenet-diagram.png]]
$$
\begin{aligned}
x_{k} &=\left[D_{k}\left(x_{k-1}\right), x_{k-1}\right] \\
&=\left[D_{k}\left(x_{k-1}\right), D_{k-1}\left(x_{k-2}\right), \ldots, D_{1}\left(x_{0}\right), x_{0}\right]
\end{aligned}
$$
- The output of each dense layer is the concatenation of the output of the previous layer, all previous layer outputs, and the input.

### Cross Stage Partial Networks
**Goal**: The main purpose of designing CSPNet is to enable this architecture to achieve ==a richer gradient combination while reducing the amount of computation==. 
<!--SR:!2024-10-31,396,208-->

The main concept is to make the gradient flow ==propagate through different network paths== by splitting the gradient flow.
<!--SR:!2024-01-20,338,268-->

**How-To**
This aim is achieved by partitioning the feature map of the base layer (inputs to the block) into two parts and then merging them through a proposed cross-stage hierarchy (one chunk of the input channels go through the block and the other chunk of the input channels are concatenated to the output of the block). 

![[cspnets-densenet-architecture.png|700]]
![[cspnet-cross-stage-densenet.png|700]]
Illustrations of (a) DenseNet and (b) our proposed Cross Stage Partial DenseNet (CSPDenseNet). **CSPNet separates the feature map of the base layer into two parts,** one part will go through a dense block and a transition layer; the other one part is then combined with transmitted feature map to the next stage.

**Result**
- The propograted gradient information can have a large correlation difference (better than [[DenseNet]]) by only passing some of the inputs to the Partial Dense Block and directly concatenating the outputs to the remaining inputs .
- CSPNet can greatly reduce the amount of computation, and improve inference speed as well as accuracy.

# Advantages
**Strengthen learning ability of a CNN**
- The lightweight versions (smaller size versions for running on less compute resources) of ResNet, ReNext, and DenseNet have sub-optimal performance compared to their original version.
- After applying CSPNet on these backbones, the computation effort can be reduced from 10% to 20% but outperforms the original backbones.

**Removing computational Bottleneck**
![[cspnet-computational-bottlenecks.png]]
The above figure shows the computational bottleneck of PeleeNet-YOLO, PeleeNet-PRN and CSPPeleeNet-EFM. The vertical lines show the computation in BFLOPs per-layer. The horizontal lines show the maximum value that any of the individual layers reach for a particular model. The CSPNet version reduces the maximum amount of computation done in a particular model. This is beneficial because the proposed CSPNet can provide hardware with a ==higher utilization rate==.
<!--SR:!2025-06-23,800,328-->

The CSPNet evenly distributes the amount of computation at each layer so the utilization rate of the hardware is more evenly spread and you don't end up with the hardware performing extra cycles or sitting idle for some layers.

Where the computational bottleneck occurs:
 - For the PeleeNet-YOLO, computational bottleneck occurs when the head integrates the feature pyramid.
 - The computational bottleneck of PeleeNet-PRN occurs on the transition layers of the PeleeNet backbone.
 - For DenseNet usually the number of channels in the base layer of each block (the inputs to the block) is much larger than the growth rate. By only passing half the channels to the block (and directly concatenating the remaining half to the outputs of the block), a partial dense block can reduce computation by almost 1/2.

**Reducing memory costs**
- CSPNet reduces memory cost via ==cross-channel pooling==. 
- In typical [[Convolutional Neural Networks#Pooling POOL|pooling layers]] you decrease the spatial size (height and width), but keep the number of channels the same (ex. HxWxC -> H/2xW/2xC). In cross-channel pooling you also decrease the number of channels (ex. HxWxC -> H/2xW/2xC/2).
- CSPNet can cut down on 75% of memory usage on PeleeNet when generating feature pyramids.
<!--SR:!2024-01-17,351,270-->

# Performance
- CSPNet achieves better accuracy at fewer BFLOPs than regular ResNext and DenseNet. It also can run faster (higher FPS).

# Related Work
- [[ResNeXt]] shows cardinality can be more effective than the dimensions of width and depth.
- [[DenseNet]] can significantly reduce the number of parameters and computations due to the strategy of adopting a large number of reuse features.
- [[SparseNet]] adjusts dense connections to exponentially spaced connections to improve parameter utilization and result in better outcomes.
- [[PRN]] (partial ResNet) shows high cardinality and sparse connections can improve the learning ability of the network by the concept of gradient combination.
- [[AI-Notes/Object Detection/Cross Stage Partial Networks]] 