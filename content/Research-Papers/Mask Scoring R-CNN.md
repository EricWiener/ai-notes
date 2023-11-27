---
tags: [flashcards]
source:
summary: an improvement on [[Mask R-CNN]] that computes mask loss using IoU of masks
---

[[Mask R-CNN]] evaluated the quality of its masks by looking at the bounding box confidence and then looking within a bounding box to see if the pixels had the correct semantic class. However, this is a poor evaluation metric if you have multiple of the same instances within a single bounding box. By improving the loss criteria, the model became better.

![[screenshot-2023-01-02_11-49-41.png]]
In the above picture, the bounding box for the orange person also includes some of the purple person. When [[Mask R-CNN]], the mask ground truth would be all "people" pixels within the box, but this also includes "people" pixels contributed by the purple mask (not just the orange mask). Mask Scoring R-CNN improves on this by just looking at the orange ground truth mask as the ground truth mask.

> [!NOTE] The [[Mask R-CNN]] mask loss just evaluates if the pixels have the correct semantic class, not the correct instance!
