---
tags:
  - flashcards
aliases:
  - Deformable transformers for end-to-end object detection
source: https://arxiv.org/abs/2010.04159
summary: a fast and efficient version of DETR that uses deformable attention and multi-scale feature maps.
publish: true
---

[Helpful YouTube Video](https://youtu.be/al1JXZTBIfU)

**The main benefits of Deformable DETR are:**
??
Reduces the memory and computational cost of [[DETR|DETR]] through deformable attention where the attention modules only attend to a small set of key sampling points around a reference. It can achieve better performance than DETR (especially on small objects) with 10x less epochs. Additionally, it can have higher resolution features since you don't need to attend to the full input.
<!--SR:!2024-02-06,127,212-->

[[Deformable Convolution]] is able to attend to sparse spatial locations with a convolution, but it lacks the ==element relation modeling of attention (all elements can relate to all other elements)==. This paper introduces [[Deformable Attention]] to attend to a small set of sampling locations as a pre-filter for prominent key elements out of all the feature pixel maps. They are able to use multi-scale features without requiring an [[Feature Pyramid Network|FPN]].
<!--SR:!2024-03-22,286,272-->

# Related Work
There are three main approaches to improving the complexity of attention:
1. Restrict the attention pattern to be a fixed local window. This results in decreased complexity, but it loses global information.
2. Learn data-dependent sparse attention. Deformable DETR uses this approach.
3. Use the low-rank property in self-attention to reduce the complexity (use linear algebra to reduce the matrix multiplies to the most significant elements).

# Deformable Attention
![[Deformable Attention#Deformable DETR 's Version of Deformable Attention]]

# Bounding Box Prediction
You start with a reference point that is used as the initial guess of the box center. The reference point is predicted as a 2D normalized coordinate ($\hat{p}_q$) using a linear project + sigmoid that takes an object query as input.

The reference point is used as the initial guess as a box center and then a detection head predicts the relative offsets w.r.t. the reference points:
$$\hat{\boldsymbol{b}}_q=\left\{\sigma\left(b_{q x}+\sigma^{-1}\left(\hat{p}_{q x}\right)\right), \sigma\left(b_{q y}+\sigma^{-1}\left(\hat{p}_{q y}\right)\right), \sigma\left(b_{q w}\right), \sigma\left(b_{q h}\right)\right\}$$

where:
- $b_{q\{x, y, w, h\}} \in \mathbb{R}$ 
- $\sigma$ and $\sigma^{-1}$ are the sigmoid and inverse sigmoid functions
- $\hat{b}_q$ will be normalized coordinates of the form $\hat{\boldsymbol{b}}_q \in[0,1]^4$. 

### Iterative Bounding Box Refinement
Each decoder layer refines the bounding boxes based on the predictions from the previous layers. Suppose there are $D$ decoder layers (e.g. $D = 6$), then given a normalized bounding box $\hat{\boldsymbol{b}}_q^{d-1}$ predicted by the ($d -1$)-th decoder layer, the $d$-th decoder layer will refine the box as:
$$\hat{\boldsymbol{b}}_q^d=\left\{\sigma\left(\Delta b_{q x}^d+\sigma^{-1}\left(\hat{b}_{q x}^{d-1}\right)\right), \sigma\left(\Delta b_{q y}^d+\sigma^{-1}\left(\hat{b}_{q y}^{d-1}\right)\right), \sigma\left(\Delta b_{q w}^d+\sigma^{-1}\left(\hat{b}_{q w}^{d-1}\right)\right), \sigma\left(\Delta b_{q h}^d+\sigma^{-1}\left(\hat{b}_{q h}^{d-1}\right)\right)\right\}$$

The initial box uses the reference point ($\hat{p}_q$) as the center with width = 0.1 and height = 0.1. Using the above notation this is:
$$\hat{b}_{q x}^0=\hat{p}_{q x}, \hat{b}_{q y}^0=\hat{p}_{q y}, \hat{b}_{q w}^0=0.1 \text {, and } \hat{b}_{q h}^0=0.1$$

To stabilize training, the gradients only back propagate through $\Delta b_{q\{x, y, w, h\}}^d$ and are blocked at $\sigma^{-1}\left(\hat{b}_{q\{x, y, w, h\}}^{d-1}\right)$.

# Two-Stage Deformable DETR
In the original DETR, object queries in the decoder are irrelevant to the current image (they are learned during training and do not change depending on the inference image). Inspired by two-stage object detectors, Deformable DETR explores a variant of Deformable DETR for generating region proposals as the first stage. The generated region proposals will be fed into the decoder as object queries for further refinement, forming a two-stage Deformable DETR.

In the first stage, given the output feature maps of the encoder, a detection head is applied to each pixel. The detection head consists of:
- A 3-layer [[Linear|FFN]] for bounding box regression.
- A linear projection for bounding box binary classification (this is similar to an objectness score - 0 means background and 1 means foreground).

Given the predicted bounding boxes from the first stage, the ==top scoring bounding boxes== are picked as region proposals. In the second stage, these proposals are fed into the decoder as the initial boxes for iterative bounding box refinement.
<!--SR:!2024-09-10,458,310-->

# Loss
Focal loss with loss weight of 2 is used for bounding box classification.

# Notation
![[Research-Papers/deformable-detr-srcs/deformable-detr-notation.png]]