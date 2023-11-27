---
tags: [flashcards]
source: [[Mask R-CNN]]
summary: an alternative to RoI Pooling introduced in Mask R-CNN to reduce quantization error.
---

[Helpful Medium Article](https://medium.com/mlearning-ai/detailed-explanation-of-roi-pooling-vs-roi-align-vs-roi-warping-8defda37461a)

RoI Align is an alternative to RoI Pool that is differentiable throughout by removing the snapping.

First you project your region proposal onto the image features. However, instead of doing snapping, we divide the projected proposal into equal sized regions. Then, within each region, you have equal sized samples (places to sample the feature map). However, these samples likely don't align to the feature map either. 

![[AI-Notes/Object Detection/roi-align-srcs/Screen_Shot 8.png]]

We can use **bilinear interpolation** to calculated a weighted average of the feature vectors from the nearest pixels to the spots we want to sample from. 

You weight each pixel by how far away it is from the spot you want to sample from. You weight the four neighbors in terms of their x-distance and y-distance, so it is a linear blend of the four. Since this is just addition, it is differentiable (unlike snapping).

![[AI-Notes/Object Detection/roi-align-srcs/Screen_Shot 9.png]]

![[Screen_Shot_6.png]]

We repeat this for every sample within all of our equally sized sub-regions. Once we compute a feature vector for each of sampled points (green dots), we can do max pooling and get a new region feature of the correct size. 

![[AI-Notes/Object Detection/roi-align-srcs/Screen_Shot 10.png]]

Now our alignment and differentiability problem is solved.