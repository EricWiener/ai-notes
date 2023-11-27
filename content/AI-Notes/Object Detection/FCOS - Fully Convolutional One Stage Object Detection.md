---
tags: [flashcards, eecs498-dl4cv]
source:
aliases: [Fully Convolutional One Stage Object Detection, FCOS]
summary: Single-stage object detector that doesn't use anchor boxes.
---

### Model
![[fcos-complete-model.png]]
- Like the [[Faster R-CNN#Region Proposal Network|Region Proposal Network in Faster R-CNN]], you pass the image through a backbone network and get a grid of features. Each feature corresponds to a point in the image.
- Then, you pass these image features through another CNN and get:
    - $C$ class scores per point in the feature map. For each point, you are giving the probability that this point lies within a bounding box corresponding to the $c$th class.
    - Additionally, you predict the ==distance to the edges== of the bounding box that this point belongs to (distance to left, right, top, and bottom).
    - You also predict a ==centerness score== for each point (a single value per feature) that says how close the point is to the center of the box. Since each point predicts a box, this is useful for determining which boxes to pay attention to.
<!--SR:!2027-06-23,1438,330!2027-06-22,1437,330-->

**Class scores per point in the feature map**
![[class-scores-per-point.png]]
- The green points should have a high score for the cat category and the red points should have a low score for all categories since they contain no objects.

**Predict coordinates of each bounding box**
![[fcos-predicting-bbox.png]]
- For each point, we predict the distance to the top (T), right (R), bottom (B), and left (L) of the bounding box it belongs to. Note: each point within a certain bounding box should predict different values since it is a different distance from the bounding box edges.
-  The bounding box prediction is category agnostic (doesn't take category into account).

**Predict centerness**
![[fcos-complete-model.png]]
- During training, each point that belongs to an image is trained to predict the corresponding class of the box and predict the box edges of the box it belongs to.
- However, this means each point is predicting its own bounding box which results in lots of box predictions. In order to figure out which of these boxes to use, the model also predicts a centerness score that ranges from 1 (center of the box) to 0 (at the edge of the box).
$$\text { centerness }=\sqrt{\frac{\min (L, R)}{\max (L, R)} \cdot \frac{\min (T, B)}{\max (T, B)}}$$
- The idea is that the CNN will likely do a better job predicting boxes for points near the center of the box. 

# Training time
### Loss:
- For the class score loss, this is treated like $C$ independent binary classification problems. Use per-category logistic regression. Logistic regression is similar to a two-category [[Softmax Loss]]. You get a binary classification score where a high score means it is likely a positive and a low score means it is likely a negative. This is applied independently per-category (unlike [[Cross Entropy Loss]] loss where you always end up with a most likely class - here you can get all classes predicted as negative).
- For the box edge predictions, you use L2 loss vs. the ground truth box. This is only applicable for points that are actually within a bounding box. Note that if a point falls into multiple boxes, you count it as belonging to the smaller ground truth box.
- For predicting centerness, you use logistic regression loss. This only applicable for positive points (points that belong to a ground truth box).


# Test time
- You pass the image through a CNN and get a set of features. You then pass these features through an additional CNN and for each point you get:
    - $C$ class scores
    - 4 values for the box edges
    - 1 value for the centerness score
- The predicted confidence for a box from each point is the product of its ==class score and centerness score==. This allows the model to damp down the confidence of predictions from points that are far away from the center of a box. The class scores for all classes are multiplied by the centerness score and then the final output boxes are chosen.
- You can then sort the bounding boxes by their predicted confidences to get your final output boxes.
<!--SR:!2024-05-12,595,330-->

**Note**: all the class scores for a particular point are multiplied by the centerness score and then the output boxes are chosen ([code here](https://github.com/tianzhi0549/FCOS/blob/07ba056d6a02db6e146514b7234e834157a80265/fcos_core/modeling/rpn/fcos/inference.py#L72)). I was initially confused by this because it seemed like it would be more efficient to just choose the top-class and then only multiply this most likely class * the centerness score to get one predicted box per-point. However, multiplying all predicted class scores by the centerness scores and then choosing final boxes allows you to have multiple objects in one point (ex. a person riding a horse or a person sitting on a bed would share the same bounding box possibly).

### Feature Pyramid
![[fcos-feature-pyramid.png]]
- FCOS uses a [[Object Detection#Feature Pyramid Network]].
- At each layer of the feature pyramid, it will make predictions (class scores, box edges, and centerness) using a shared head (same weights for each layer). You can then choose the final output bounding boxes from all the boxes predicted by the shared head.