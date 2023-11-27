---
tags: [flashcards, eecs442, eecs498-dl4cv]
source:
summary: assigns class labels to pixels in an image (with no distinction between different entity instances).
---
Note: this is called semantic segmentation because:
- Segmentation: this is a general task of breaking up the image into chunks
- Semantic: we are assigning class labels to the chunks


# Approaches to per-pixel classification
![[AI-Notes/Image Segmentation/image-segmentation-srcs/Screen_Shot 1.png]]

For every pixel, we want to say what category that pixel belongs to.

## Idea #1: what's the object class of the center pixel
You can apply a sliding window over the image. For each window, try to classify it and assign the center pixel the label of the window.

![[AI-Notes/Image Segmentation/image-segmentation-srcs/Screen_Shot 2.png]]

This would be **very slow**, since you would need a window for every pixel. No one does this in practice. This doesn't reuse shared features between overlapping patches.

## Idea #2: Fully Convolutional Networks
![[semantic-segmentation-fully-conv.png]]
We can use a convolutional neural network to make predictions for the entire image at once. You have a $3 \times H \times W$ input and then predict a $C \times H \times W$ output. Each pixel has a corresponding distribution over the $C$ classes. You can then use a per-pixel cross-entropy loss during training.

### Benefit of a fully convolutional network (vs. using linear layers):
The convolutional layers scale to any image size. Since the CONV layers are just filter weights, it doesn't matter what the size input is. You can just apply the filter.

The reason the input size is restricted in CNN's is because the FC layers at the end are a fixed size, so the input needs to be a fixed size in order to match.

The fully convolutional network gets rid of the FC layers so you can now handle any size input. As long as you use same padding, you will end up with a matrix of size $C \times H \times W$ where $C$ is the number of possible labels and $H \times W$ is the original dimensions of the image. Now, you can look at each label and see what the probability is that a certain pixel belongs to that category. You can compute the argmax over this matrix and get per-pixel classification.

### Problem
We **can't downsample the image** because we need the output predictions to be the same resolution as the input image.

**Problem #1**: without downsampling, the effective receptive field is linear in the number of conv layers. With L $3 \times 3$ layers, the receptive field is $1 + 2L$. You will need many layers in order to get a large enough receptive field to make good predictions.

**Problem #2**: convolution on high resolution images is expensive. In modern architectures, like [[ResNet]] the input image is aggressively downsampled to improve computation. Without downsampling, the feature maps (outputs after applying the filters) will be much larger and take up more memory.

We can solve these issues if we **downsample the image and then upsample it again.** 

### Idea #3: [[Dilated Convolution]]
Dilated Convolutions allow us to get a larger receptive field (by spreading out the kernel) while maintaining the same number of parameters.

However, even with dilated convolutions, you will likely still need to downsample from the original resolution to avoid having huge feature maps, so this will result in the output being at a smaller resolution than the inside. You **need a way to upsample** so that we can go from a lower resolution back up to the original resolution.

### Idea #4: Fully Convolutional Networks with downsampling + upsampling
This is a form of an [[Encoder-Decoder Models|Encoder-Decoder]] network. Encoder-decoder models are used when you need a very high resolution output. You have an encoder that makes the image smaller. Then you have a second step that takes you back up in size.

![[fully-conv-w-downsample-upsample.png]]
- We can design the network to first downsample the image and later upsample the image.
- This allows us to perform computation with a smaller spatial size for most of the network, but still predict at the original resolution.
- You can then use per-pixel cross-entropy loss.

**Downsampling**: we can use pooling or strided convolution.

**Upsampling**:
- [[Upsampling]]: you can use bed of nail, nearest neighbor, **bilinear**, or bicubic upsampling.
- [[Max Unpooling]]: this is a older (2015) version of upsampling that **isn't used anymore**. It pairs max pooling for downsampling with a corresponding upsampling layer.
- [[Transposed Convolution]]: this is an upsampling layer that includes learned weights.

**Example papers**:
- [[Fully Convolutional Networks for Semantic Segmentation]]
