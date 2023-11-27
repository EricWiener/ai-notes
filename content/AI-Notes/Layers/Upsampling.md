---
tags: [flashcards]
source:
aliases: [unpooling, bilinear interpolation, bicubic interpolation]
summary: upsampling is used to increase spatial dimensions. There are multiple approaches for how to do this.
---

### Bed of Nails Upsampling
![[upsampling-bed-of-nails.png|300]]
Bed of nails upsampling just places each input value in the top-left position of the larger region and puts zeros everywhere else. 

This isn't used that much in practice because of issues with aliasing.


### Nearest Neighbor Upsampling
![[upsampling-nearest-neighbor.png|300]]
Nearest neighbor upsampling just duplicates the input values everywhere into the larger grid.

### Bilinear Interpolation Upsampling
![[upsampling-bilinear-interpolation.png|300]]
You can also use **bilinear interpolation**  to fill out the larger grid. You choose the closest neighbors in $x$ and $y$ and compute a linear combination of these points. This is differentiable.

### Bicubic interpolation Upsampling
![[upsampling-bicubic-interpolation.png|300]]
Bicubic interpolation is an extension of **bilinear interpolation** that just uses the three closest neighbors in x and y to construct cubic approximations. This is what is usually used to resize images in your web browser. However, it is less common than bilinear interpolation for neural networks since it is slower .



