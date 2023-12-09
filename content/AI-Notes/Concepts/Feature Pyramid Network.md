---
tags:
  - flashcards
aliases:
  - FPN
source: https://arxiv.org/abs/1612.03144
summary: a feature pyramid network makes use of preceding feature maps at multiple scales to make predictions (ex. BBOX or masks) at multiple resolutions.
publish: true
---
![[encoder-decoder-architecture.png|350]] 

We can use a feature pyramid network to resize our feature map to different sizes. We don't want to down-size the image and then run it through the CNN because then we will have ==less features==. Additionally, we can't just use the outputs of different layers of the CNN to get different size features because the ==earlier layers== won't do as good a job of capturing information. 
<!--SR:!2029-11-24,2179,350!2023-12-08,480,330-->

The solution is to add connections that feed information from high level features (from the backbone) to lower level features. Now, **all the levels benefit from the backbone**, but you still operate at different resolutions. The FPN gives us a ==set of feature maps== of varying spatial resolutions coming out of the backbone network (vs a single feature map).
<!--SR:!2027-07-13,1494,343-->

![[concrete-example-feature-pyramid.png|350]]

In the example above, stage 5 of the backbone produces a 7x7 feature map. These features are passed to an object detector. They are also upsampled and combined with features from the stage 4 map (the features from stage 4 are passed through a 1x1 conv first). The object detector is again run at this level and this is repeated for all stages of the network.

For [[Faster R-CNN]], the detector at each level gets its own region proposal network to produce proposals per-level (each RPN operates independently for each level, but uses the same weights). The proposals from all levels are all then combined and passed to a shared second stage (per-region head that predicts classification and bounding box regression).

#### Feature Pyramid Network vs. Image Pyramid
![[image-pyramid-comparison.png|400]]

In an image pyramid, you just resize the raw image to different scales and then make predictions for these different sizes. You aren't downsizing the image via a network and are instead just downsizing the raw image (and losing information along the way).

#### Feature Pyramid Network vs. [[Encoder-Decoder Models|U-Net]] Architecture
![[unet-architecture.png|400]]

In an U-Net Architecture, you just predict at the final layer of the decoder. In a FPN, you have a corresponding prediction for each stage in the backbone. Additionally, you have a 1x1 conv between the layers in the backbone and the features you make predictions on (vs. just having an additive skip connection in the U-Net Architecture).