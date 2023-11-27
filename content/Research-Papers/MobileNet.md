---
tags: [flashcards]
source: [[MobileNet.pdf]]
summary:
---

- Very efficient models (small, low latency) for use in mobile and embedded vision applications
- Models are defined with two hyper-parameters: width multiplier and resolution multiplier
- Most approaches for small models are either compressing pretrained networks or training small networks directly.
- MobileNets primarily focus on optimizing for latency but also yield smaller networks.

# Depthwise Convolutions
- Built primarily from [[Depthwise Separable Kernels]].
- **Depthwise separable convolutions factorize a standard convolution into a depthwise convolution and a 1x1 convolution called a pointwise convolution.**
- A standard convolution **both filters and combines** inputs into a new set of outputs in one step. The depthwise separable convolution splits this into two layers, a **separate layer for filtering and a separate layer for combining**. This factorization has the effect of drastically reducing computation and model size.

**Computational Cost of Standard Conv Layer**

- Input: $\mathrm{D}_F \times \mathrm{D}_F \times \mathrm{M}$ where $\mathrm{D}_F$ is the spatial width and height (square input) and $M$ is the number of channels.
- Output: $\mathrm{D}_F \times \mathrm{D}_F \times \mathrm{N}$ same spatial dimensions with $N$ output channels
- Kernel size: $\mathrm{D}_K \times \mathrm{D}_K \times \mathrm{M} \times \mathrm{N}$ where $\mathrm{D}_F$ is the spatial dimension of the square kernel, $M$ is the number of input channels, and $N$ is the number of output channels. The kernel is applied with stride one and same padding.
- Standard convolution cost: $(\mathrm{D}_K \times \mathrm{D}_K \times \mathrm{M} \times \mathrm{N}) \times (\mathrm{D}_F \times \mathrm{D}_F)$
    - This is the number of elements in the kernel (note that the kernel includes the number of output channels as a fourth-dimension vs. considering $N$ three-dimensional kernels) times the number of spatial elements in the output.

> [!note]
> The computational cost for a standard conv layer depends multiplicatively on the number of input channels M, the number of output channels N the kernel size $\mathrm{D}_K \times \mathrm{D}_K$ and the feature map size $\mathrm{D}_F \times \mathrm{D}_F$
> 

**Depthwise Separable Convolutions**

- Depthwise convolutions to apply a single filter per each input channel (input depth). It has a computational cost of $D_{K} \cdot D_{K} \cdot M \cdot D_{F} \cdot D_{F}$
- Depthwise convolution is extremely efficient relative to standard convolution. However it only filters input channels, it does not combine them to create new features.
- Pointwise convolution is used to create a linear combination of the output of the depthwise layer to generate new features.

**Depthwise Seperable Convolution Cost**

The sum of the depthwise and 1x1 pointwise convolution cost is given by:

$$
D_{K} \cdot D_{K} \cdot M \cdot D_{F} \cdot D_{F}+M \cdot N \cdot D_{F} \cdot D_{F}
$$

This is a reduction in cost from the regular convolution (denominator) of:

$$
\frac{D_{K} \cdot D_{K} \cdot M \cdot D_{F} \cdot D_{F}+M \cdot N \cdot D_{F} \cdot D_{F}}{D_{K} \cdot D_{K} \cdot M \cdot N \cdot D_{F} \cdot D_{F}} = \frac{1}{N}+\frac{1}{D_{K}^{2}}
$$

**MobileNet’s use of Depthwise Separable Convolutions**
![[standard-conv-vs-depthwise-seperable.png]]
- MobileNet uses batchnorm and ReLU for both layers.
- MobileNet uses 3 × 3 depthwise separable convolutions which uses between 8 to 9 times less computation than standard convolutions at only a small reduction in accuracy as seen in Section

# Network Structure + Training

- First layer which is a full convolution
- Final fully connected layer which has no nonlinearity and feeds into a softmax layer for classification
- Down sampling is handled with strided convolution in the depthwise convolutions as well as in the first layer. A final average pooling reduces the spatial resolution to 1 before the fully connected layer.
- Puts nearly all of the computation into dense 1 × 1 convolutions (95% of computation time and 75% of parameters).
- Less regularization and data augmentation techniques because small models have less trouble with overfitting

### Width multiplier ($\alpha$)

- The role of the width multiplier α is to thin a network uniformly at each layer
- The width multiplier scales up/down the number of input and output channels. The number of input channels M becomes $\alpha M$ and the number of output channels N becomes $\alpha N$.
- Width multiplier has the effect of reducing computational cost and the number of parameters quadratically by roughly $\alpha^2$

### Resolution Multiplier ($\rho$)

- This will scale up/down the height and width of the input (which in turn affects all subsequent layers)
- It is a scalar between 0 and 1 and is typically set so the input has a resolution of 224, 192, 160, or 128.
- Total computational cost is now $D_{K} \cdot D_{K} \cdot \alpha M \cdot \rho D_{F} \cdot \rho D_{F}+\alpha M \cdot \alpha N \cdot \rho D_{F} \cdot \rho D_{F}$
- Has the effect of reduction computational cost by $\rho^2$

# Results

- Using depthwise separable convolutions compared to full convolutions only reduces accuracy by 1% on ImageNet was saving tremendously on mult-adds and parameters
- Accuracy drops off smoothly as the width multiplier is decreased until the architecture is made too small at $\alpha = 0.25$
- Making MobileNets thinner (decrease layers per channel) is 3% better than making them shallower (decrease number of layers).
- Accuracy drops off smoothly across resolution.
- MobileNet is nearly as accurate as VGG16 while being 32 times smaller and 27 times less compute intensive. It is more accurate than GoogleNet while being smaller and more than 2.5 times less computation.

**Distillation**

- Synergistic relationship between MobileNet and distillation [9], a knowledge transfer technique for deep networks.
- Distillation [9] works by training the classifier to emulate the outputs of a larger model2 instead of the ground-truth labels, hence enabling training from large (and potentially infinite) unlabeled datasets.
- Marrying the scalability of distillation training and the parsimonious parameterization of MobileNet, the end system not only requires no regularization (e.g. weight-decay and early-stopping), but also demonstrates enhanced performances.
- Resilient to aggressive model shrinking

# Other work

- [[Depthwise Separable Kernels]] were introduced in 26
- [[Flattened Convolutional Neural Networks for Feedforward Acceleration]] build a network out of fully factorized convolutions (16). [[Notes]]
- Factorized Networks introduces factorized convolutions + topological connections (34)
- Xception network uses depthwise separable filters (3)
- Distillation uses larger networks to teach a smaller network (9)
- Transform networks (28)
- [[Deep Fried Convnets]]