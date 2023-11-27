---
tags: [flashcards]
source:
summary: IoU quantifies differences between bounding boxes.
aliases: [intersection over union, jaccard similarity, jaccard index]
---
**Intersection over union (IoU):** this is used for object detection to measure how well the predicted ==bounding box matches the ground truth bounding box==. You divide the area of the region where the bounding boxes overlap by the total area of both bounding boxes.
<!--SR:!2024-06-08,622,330-->

![[iou-diagram.png]]

If the bounding boxes overlap perfectly, you will have an IoU of 1. If they don't overlap at all, you will have an IoU of 0.

This is also sometimes called **Jaccard similarity.**

### Interpreting IoU
- IoU > 0.5 is "decent"
- IoU > 0.7 is "pretty good"
- IoU > 0.9 is "almost perfect"

![[iou-0.5.png|300]]
![[iou-0.72.png|300]]
![[iou-0.91.png|300]]