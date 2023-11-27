---
tags: [flashcards, eecs498-dl4cv]
source:
summary: a single stage object detector that is similar to the RPN from Faster R-CNN.
---

This is a single stage object detector. This is similar to the [[Faster R-CNN#Region Proposal Network|Region Proposal Network from Faster R-CNN]]. However, unlike in [[Faster R-CNN]], you don't need to first predict regions, crop the regions, and then predict classifications per-region. Now, you predict the classifications and bounding boxes in one go.

![[retina-net-diagram.png]]

### Model Overview:
- Start with input image.
- Run input image through a backbone to extract image features.
- Each feature in the feature grid corresponds to a point in the input image.
- You then consider a set of anchor boxes centered at each of these points.
- For each anchor box:
    - Predict the class of the anchor box. However, these anchors are now category-specific (vs. just being object/not object). We are predicting $C + 1$ categories for each anchor box ($C$ classes + 1 for background).
    - Predict bounding box transform

### Class Imbalance
- There are a lot more background anchors than non-background anchors. This would cause the model to learn to just always predict background (using cross-entropy loss doesn't work great for an imbalanced class distribution).
- In order to handle this, the paper introduced a new loss function called [[Focal Loss]].

### [[Object Detection#Feature Pyramid Network|Feature Pyramid Network]]
![[retina-net-feature-pyramid.png]]
RetinaNet uses a FPN in the same way that Faster R-CNN does. You have anchors at multiple scales of the feature pyramid network and then predict class labels and bounding box regression for the anchors at different scales.

### Performance
![[retina-net-performance-chart.png]]
- At the same performance (COCO AP) as Faster R-CNN, RetinaNet is faster.
- At the same inference time as Faster R-CNN, RetinaNet has better performance.
- You can trade off between speed and inference time for a single stage detector by adjusting:
    - The backbone size
    - The input image resolution.
    - Note: for a two-stage detector you can additionally adjust the number of proposals. This isn't applicable for a single stage detector since you don't generate proposals as a seperates step.