---
tags: [flashcards]
aliases: [Single-Shot Image Parser]
source: https://arxiv.org/abs/1902.05093
summary: precursor to Panoptic-DeepLab and introduces bootstrapped cross-entropy loss.
---

[[DeeperLab_ Single-Shot Image Parser, Tien-Ju Yang et al., 2019.pdf]]

### Weighted Bootstrapped Cross-Entropy Loss
The semantic segmentation prediction is trained to minimize the **bootstrapped cross-entropy loss** which sorts the pixels based on [[Cross Entropy Loss]] loss and only backpropagates the errors in the top-K positions.

$K$ is set to $K = 015 \cdot N$ where $N$ is the total number of pixels in the image. This will make it so you only backpropagate the loss for the worst 15% of pixels in the image.

The weighted boostrapped cross-entropy loss is given by:
![[Annotated Weighted Bootstrap Cross Entropy Loss]]
- $N$ is the total number of pixels in the image
- $i$ will iterate over all pixels in the image
- $y_i$ is the target class label for pixel $i$
- $p_{i, j}$ is the predicted probability for pixel $i$ and class $j$. This means $p_{i, y_i}$ is the predicted probability for the target class for pixel $i$.
- $\mathbb{1}[x]$ is the indicator function that is $1$ is $x$ is true and 0 otherwise. $t_K$ is a threshold set so that only the pixels with the top-K highest losses will make $\mathbb{1}\left[p_{i, y_i}<t_K\right]$ true.
- $w_i$ is set to 3 for pixels that belong to small instances (area smaller than $64 \times 64$) and 1 otherwise. This is the "weighted" part of "weighted boostrapped cross-entropy." Using this weight ensures the network focuses on both hard pixels and small instances.

