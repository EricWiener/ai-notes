---
tags: [flashcards]
source: [[ResNeXt.pdf]]
summary: Increasing cardinality (splitting input along channel dimension and operating on it using multiple equivalent parallel pathways) showed improvements over traditional ResNet.
---

[https://medium.datadriveninvestor.com/resnext-explained-part-2-b65efc5a4adc](https://medium.datadriveninvestor.com/resnext-explained-part-2-b65efc5a4adc)

[https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215](https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215)

> [!note]
> This paper refers to number of layers as **depth,** number of channels in a kernel as **width,** and number of pathways/groups as **cardinality.** The height/width (in traditional sense) are referred to as the **spatial map.**
> 
![[resnext-cardinality-intro.png]]
- This strategy exposes a new dimension, which we call “cardinality” (the size of the
set of transformations), as an essential factor in addition to the dimensions of depth and width.
- Even with the same complexity, increasing cardinality can improve classification accuracy and is more effective than going deeper or wider.
- Using a consistent design can reduce the risk of over-adapting hyper-parameters to a specific dataset.
- Module in the network performs a set of transformations, each on a low-dimensional embedding, whose outputs are aggregated by summation.
- Uses the idea from AlexNet for [[Grouped Convolutions]] (though AlexNet did this to split the model to multiple GPUs).

### Reference to [[Going Deeper with Convolutions|GoogleNet]]
In an Inception module, the input is split into a few lower-dimensional embeddings (by 1×1 convolutions), transformed by a set of specialized filters (3×3, 5×5, *etc*.), and merged by concatenation. It can be shown that the solution space of this architecture is a strict subspace of the solution space of a single large layer (*e.g*., 5×5) operating on a high-dimensional embedding.

![[inception-module.png]]

> [!note]
> The ResNeXt paper is saying that transforming the input into a **lower-dimensional** embedding and then transforming is a strict subspace of the solution space of a single large layer operating on a **high dimensional embedding.** The key part is that a lower-dimensional embedding loses information from the higher dimensional embedding.
> 

The split-transform-merge behavior of Inception modules is expected to approach the representational power of large and dense layers, but at a considerably lower computational complexity.

- Filter number and sizes in each inception module are tailored for each individual transformation and the modules are customized stage-by-stage. This makes it **unclear how to adapt the Inception architecture to new datasets/tasks.**

### Reference to [[Deep Roots Improving CNN Efficiency with Hierarchical Filter Groups]]

Decomposition is a widely adopted technique to reduce redundancy of deep convolutional networks and accelerate/compress them. Ioannou *et al*. [16] present a “root”-patterned network for reducing computation, and branches in the root are realized by [[Grouped Convolutions]].

> [!note]
> Deep Roots focused more on reducing computation and getting around the same accuracy while ResNeXt kept computation the same and tried to get better accuracy.
> 

### Model Architecture

![[model-architecture.png]]

Inside the brackets are the shape of a residual block, and outside the brackets is the number of stacked blocks on a stage. “C=32” suggests grouped convolutions with 32 groups. **The right column is the important column.**

Each time when the spatial map is downsampled by a factor of 2, the width of the blocks is multiplied by a factor of 2. The second rule ensures that the computational complexity, in terms of FLOPs (floating-point operations, in # of multiply-adds), is roughly the same for all blocks.

> [!note]
> Note that “width” refers to the number of channels within each filter.
> 

### Relation to Simple Neurons

Neurons in neural networks perform **inner product** (weighted sum). This is just scaling each element and then adding them up (shown below).

$$
\sum_{i=1}^{D} w_{i} x_{i}
$$

This operation can be re-interpreted as splitting, transforming, and aggregating:

1. *Splitting*: the vector x is sliced as a low-dimensional embedding, and in the above, it is a single-dimension subspace $x_i$.
2. *Transforming*: the low-dimensional representation is transformed, and in the above, it is simply scaled: $w_ix_i$
3. *Aggregating*: the transformations in all embeddings are aggregated by $\sum_{i=1}^{D}$

**This paper builds on the idea of the simple neuron:**

$$
\mathcal{F}(\mathbf{x})=\sum_{i=1}^{C} \mathcal{T}_{i}(\mathbf{x})
$$

- Changes the transform from a simple scaling to an arbitrary function $\mathcal{T}_{i}(\mathbf{x})$ that transforms $x$ into an (optionally lower dimensional) embedding and then transforms it.
- Instead of using the number of elements in the incoming vector ($D$), they use $C$ as the set of transformed inputs to be aggregated. $C$ is referred to as **cardinality.**
- This paper uses identical $\mathcal{T}_{i}$’s that are all BottleNeck blocks where the first $1 \times 1$ layers produce low-dimensional embeddings.
- **BottleNeck Block**
- Aggregates transformations by adding up transformed inputs + residual skip connection: $\mathbf{y}=\mathbf{x}+\sum_{i=1}^{C} \mathcal{T}_{i}(\mathbf{x})$

### Skip Connection
![[resnext-20220720064022612.png]]

- The ResNet architecture uses skip connections via addition to preserve the gradient (prevent vanishing gradients) and allow information from previous layers to flow to later layers.


### Equivalent Formulations
![[equivalent-building-blocks.png]]
All of the above forms are equivalent forms of the building block of ResNeXt

1. This is the building block actually used. Note that the final layer before summing all paths together is a 1x1 kernel that transforms from 4 channels to 256 channels. You then end up with 32 pathways with 256 channels that are then added together.
2. Here, you concatenate after the pathways go through their 3x3 kernels with 4 output channels each. You then concatenate and then pass the entire 128 channel tensor through a 1x1 kernel to produce 256 channel output.
3. This is the same as the left-most, but you use grouped convolutions instead of splitting up pathways.

### Relationship between cardinality and width
![[cardinality-and-width.png]]
- The paper refers to the bottleneck width (number of channels) as $d$. Note that the bottleneck width doubles each time the spatial map is divided by 2 (for every subsequent stage), so it is referred to as a variable.

### Results
- With cardinality $C$ increasing from 1 to 32 (and the same complexity), the error rate keeps reducing. There was also **lower training error suggesting the gains are not from regularization (reducing overfitting), but from stronger representation.**
- Increasing cardinality (and allowing complexity to increase) shows much better results than going deeper or wider.
- Using residual connections (connect input to the output of the block) showed a benefit.