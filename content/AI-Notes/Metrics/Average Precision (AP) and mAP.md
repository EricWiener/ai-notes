---
tags:
  - flashcards
  - eecs498-dl4cv
source: https://pro.arcgis.com/en/pro-app/latest/tool-reference/image-analyst/how-compute-accuracy-for-object-detection-works.htm
aliases:
  - mean average precision
  - mAP
  - average precision
  - COCO mAP
summary: the average of the average precision for each class.
publish: true
---
![[precision-recall-example-curve.png]]

**Average precision** is calculated by finding the ==area== under a [[Precision-Recall Curve]]. Precision and recall are always between 0 and 1. Therefore, AP falls within 0 and 1 also. 
<!--SR:!2023-12-29,489,330-->

**Mean average precision** (mAP) is just the mean average precision. You first calculate the average precision for each class the detector can predict and then take the average of all those averages.

Note: you pronounce mAP as m-a-p (not as "map").

### Average precision (AP)
1. Run the object detector on all test images.
2. Drop overlapping bounding boxes with [[Non-maximum Suppression]].
3. Plot a [[Precision-Recall Curve#Explained through Object Detection POV|precision-recall curve]] where the threshold is based on the minimum IoU overlap.
4. After plotting all the precision-recall points, we can calculate the area under the curve. 0.0 means we did terribly. 1.0 means we did really well. 

![[precision-recall-ap.png]]

To get AP = 1.0, you need to hit all ground truth boxes with [[IoU]] > 0.5 and have no "false positive" detections ranked above any "true positives." This is very hard to get. Note: you can have false positive detections, but as long as they are ranked below the true positives and you have already covered all the GT boxes, they will just all be points on the line $x = 1.0$ and your area will still be 1.

We use this complicated metric because it allows you to make trade-offs between how many objects you correctly classify and how many objects you incorrect classify. Self-driving cars would prefer not to miss any objects. By computing the average precision metric, it summarizes all the points along the precision-recall curve.

### Mean Average Precision (mAP)
Average precision is calculated per-class. Mean average precision just averages the AP for each category.

### COCO mAP:
When computing the mAP, we calculated the average precision for each class using the criteria that a predicted bounding box matched a ground truth box if it had IoU > 0.5. For COCO mAP, you calculate the mAP with ==different IoU thresholds== and then average these. You can consider thresholds like 0.5, 0.55, 0.5, ..., 0.95.
<!--SR:!2026-08-27,1173,310-->

### mAP for segmentation 
The only difference between mAP for object detection and instance segmentation is that when calculating overlaps between predictions and ground truths, one uses the pixel-wise IOU rather than bounding box IOU.