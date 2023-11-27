---
tags: [flashcards, eecs498-dl4cv]
source:
summary: more complicated version of upsampling that isn't used anymore that pairs max pooling for downsampling and max unpooling for upsampling.
---
This is a more complicated version of [[Upsampling]].

![[AI-Notes/Image Segmentation/image-segmentation-srcs/Screen_Shot 9.png]]

The unpooling operation is now paired with a corresponding downsampling operation that took place earlier in the network. When you do a max pooling to downsample, you will remember the position of the maximum value.

When you do unpooling, you will do something similar to [[Upsampling#Bed of Nails Upsampling|bed of nails unpooling]], but instead of just always placing the value into the top-left corner, you will place it into the position where the max value was during downsampling.

The idea is that we will ideally end up with better alignment between feature vectors if we pool and unpool in the same way. This helps avoid small spatial shifts in the network. When you downsample with max pooling, it will shift the center of the receptive field. Max unpooling will unshift the receptive field.

> [!note]
> If you used average pooling to downsample, you should probably use bicubic or bilinear to unpool. If you used max pooling, you should probably use max unpooling.
> 
