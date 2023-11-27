---
tags: [flashcards]
source:
summary:
---

Fast R-CNN computes the features for the entire image at once. Then, the bounding boxes use the already extracted features. This saves you from having to pass each proposed region through a CNN (especially you will likely end up with overlap between multiple proposed regions since ~2000 proposals are generated for each image).

![[screenshot-2022-03-21_07-35-50.png]]

Now, we compute the CNN features for the entire image. This CNN is called the "**backbone**" of the network and can be a model such as AlexNet, VGG, ResNet, etc. 

Then, we generate region proposals based on the original image (original RGB values). We then align the region proposals over the features. We extract these features and re-size them to be square. Then we pass these squares to a lightweight network (could be [[Linear|mlp]], [[Convolutional Neural Networks|CONV layer]], etc.) which gives us predictions (this operates independently on each region).

Fast R-CNN is much faster than "Slow" R-CNN because ==most of the computation happens in the backbone network and can be shared==.
<!--SR:!2024-11-03,613,330-->

During test time for R-CNN, close to 90% of the computation is taken up by region proposals. These proposals are done using **selective search** on CPU. 

### Fast R-CNN and transfer learning
![[fast-rcnn-transfer-learning.png]]

You can use a pre-trained model and then modify it for R-CNN. You use most of the model as the backbone and then initialize the per-region network with the final layers.

![[ROI Pooling]]

![[RoI Align]]