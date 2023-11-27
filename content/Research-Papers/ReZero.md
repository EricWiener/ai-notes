---
tags: [flashcards]
source: https://paperswithcode.com/method/rezero
summary: normalization approach for transformers that initializes each layer to perform the identity operation.
---

**ReZero** is a [normalization](https://paperswithcode.com/methods/category/normalization) approach that dynamically facilitates well-behaved gradients and arbitrarily deep signal propagation. The idea is simple: ReZero initializes each layer to perform the identity operation. For each layer, a [residual connection](https://paperswithcode.com/method/residual-connectio) is introduced for the input signal $x$ and one trainable parameter $\alpha$ that modulates the non-trivial transformation of a layer $F(x)$:

$$\mathbf{x}_{i+1}=\mathbf{x}_{i}+\alpha_{i} \boldsymbol{F}\left(\mathbf{x}_{i}\right)$$

where $\alpha = 0$ at the beginning of training. Initially the gradients for all parameters defining $F$ vanish, but dynamically evolve to suitable values during initial stages of training. The architecture is illustrated in the figure below.

![[rezero-diagram.png]]


### ReZero vs. FixUp
- Fixup initialized the weight matrix to zero and added a scale parameter (scalar multiplier), rezero added a scalar multiplier but initialized it to zero and not weight matrix itself.
- Fixup also dealt with initialization for skips over multiple "blocks", in which case some of them were not initialized to zero.
- Fixup focused on getting rid of batchnorm, while this paper kept batchnorm for the regularization purposes, and did a comparison with Layer-Norm. Fixup solved the regularization problem with an aggressive implementation of Mixup.
- This paper focused on reducing the training speed, which I don't remember from fixup.
- Theoretically speaking, rezero takes more complex (advanced?) theories to consideration. Fixup was more about keeping the variance in check while here the signal propagation in both ways are considered.
[Source](https://www.reddit.com/r/MachineLearning/comments/fh0bp6/comment/flg5yhs/?utm_source=share&utm_medium=web2x&context=3)

### ReZero vs. SkipInit
SkipInit (De and Smith, 2020), an alternative to the BatchNorm, was proposed for ResNet architectures that is similar to ReZero. The authors find that in deep ResNets without BatchNorm, a scalar multiplier is needed to ensure convergence. We arrive at a similar conclusion for the spe- cific case considered in (De and Smith, 2020), and study more generally signal propagation in deeper networks across multiple architectures and beyond BatchNorm.