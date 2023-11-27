---
tags: [flashcards, eecs498-dl4cv]
source: [[Mask R-CNN, Kaiming He et al., 2017.pdf]]
summary: instance segmentation is able to predict segmentation masks and identify individual entities.
---
[[Mask R-CNN, Kaiming He et al., 2017.pdf]]

![[instance-segmentation-example.png]]

**Motivation**:
Depending on the shape of the object, a bounding box might not tell you much (think a bounding box for a sentence on a bent page). We can use segmentation to give us more information about where the entity actually is within the bounding box.

**Approach**:
Instance segmentation can be done by first performing object detection with bounding boxes and then predicting a segmentation mask for each object. The mask can be predicted as a binary mask (0 or 1 depending on if the pixel is in the image).

**Architectures**:
[[Mask R-CNN]] is one example of an instance segmentation network.

# Propsal-based vs. FCN-Based
In proposal-based method you first generate bounding boxes and then generate a mask within each bounding box. An alternative approach (FCN-based in diagram below) is to first predict a semantic segmentation and then group the "things" into individual instances.
![[panoptic-deeplab-20230102110621454.png]]

[[Mask R-CNN]] took a different approach and outputs class-agnostic masks and a bounding box regression + class label in parallel for each region of interest.