---
tags: [flashcards]
source: [[Deep Sets, Manzil Zaheer et al., 2017.pdf]]
summary: a new type of architecture for dealing with sets as inputs to models (ex. point cloud or grouping objects).
---

# Abstract
- This paper looks at problems where the output doesn't depend on the order of the inputs ([[Invariance vs Equivariance|permutation invariant]]).
- They design a network architecture that can operate on sets and describe the characteristics that functions need to have to be **permutation invariant**.
- They also derived the necessary and sufficient conditions for permutation equivariance in deep models (they had two variant of their model - one was invariant and one was equivariant).
- Achieve decent performance on statistic estimation, point cloud classification, set expansion, and outlier detection.

# Introduction
- In supervised learning, we have an output label for a set that is invariant or equivariant to the permutation of set elements. Ex: given a pointcloud of an object, the object remains the same no matter how you list the points.
- In unsupervised learning, the set structure needs to be learned. One example is given a set of similar objects (ex: {lion, tiger, cheetah}), find new objects from a large pool of candidates that are similar (ex: jaguar or leopard).
- We propose a fundamental architecture, DeepSets, to deal with sets as inputs

# Permutation Invariance and Equivariance
See: [[Invariance vs Equivariance]].
- Invariant: for any permutation $\pi$: $f\left(\left\{x_{1}, \ldots, x_{M}\right\}\right)=f\left(\left\{x_{\pi(1)}, \ldots, x_{\pi(M)}\right\}\right.$. This means you get the same output regardless of how you permute your input.
- Equivariant: you change the order of the inputs and the output order changes equally.

# Deep Sets
- To maintain permutation invariance, the key is to ==add== up the input representations and then apply nonlinear transforms.
<!--SR:!2024-09-27,576,310-->

# Applications and Results
- The permutation invariant model was applied to sum of digits, classification of point clouds, etc.
- The permutation equivariant model was applied to outlier detection (given a set of images, which doesn't belong). 
- The model didn't set any records, but it did well compared to SOTA without being specialized for a particular task.