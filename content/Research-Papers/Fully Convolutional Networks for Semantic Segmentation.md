---
tags: [flashcards]
aliases: [FCNs, FCN]
source: https://ar5iv.labs.arxiv.org/html/1411.4038
summary: uses fully convolutional networks (1x1 conv instead of MLP) for semantic segmentation
---

> [!NOTE] Reasoning for using skip-connections to combine features from earlier layers with later layers.
> As they see fewer pixels, the finer scale predictions should need fewer layers, so it makes sense to make them from earlier feature maps. Combining fine layers and coarse layers lets the model make local predictions that respect global structure.

### [[Transposed Convolution|Deconvolution]]
The paper makes use of deconvolution layers (transposed convolution) to increase the spatial resolution to the original image size.

Simple upsampling (ex. with bilinear interpolation) with factor $f$ is convolution with a fractional input stride of $1/f$. For any integer $f$, you can also implement this with a backwards convolution (deconvolution) with an output stride of $f$. Such an operation is trivial to implement, since it simply reverses the forward and backward passes of convolution. 

The paper initializes the upsampling layers to bilinear interpolation, but allow the parameters to be learned.
