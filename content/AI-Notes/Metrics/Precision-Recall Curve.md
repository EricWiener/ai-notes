---
tags: [flashcards]
source:
aliases: [precision, recall]
summary: a precision-recall curve compares the precision and recall of a model at different confidence thresholds.
---

- Precision tells us ==true positive / total predicted positive==. If you want a model with high precision, you want to predict fewer boxes with higher confidence.
- Recall tells us ==true positive / total actual positive==. If you want a model with high recall, you want to predict more boxes even if some of them are wrong. You can remeber recall because it is TP/**r**eal positive
<!--SR:!2024-12-30,720,270!2028-08-13,1774,328-->

To remember this, you can use: each is the true positive (TP) as the numerator divided by a different denominator.
-   **P**recision: TP / **P**redicted positive
-   **R**ecall: TP / **R**eal positive

predicted and correct non-moving / correct non-moving

# Precision-Recall Curve
![[pr-curve-example.png|300]]

When we have a classifier that predicts a confidence score for classes, we can decide what cut-off threshold to use for each class. Each class would have its own precision-recall curve. You then plot for each threshold you are considering, the precision and recall. 

If you have a 99% threshold (you must be super confident), then your precision will go to 1 because the things you predict positive are all correct. However, you will likely miss most of the actual positive instances and your recall will be low. 

A perfect classifier will have a precision of 1.0 and recall of 1.0. The ideal threshold is the point along the precision-recall curve closest to the top-right.

# Explained through Object Detection POV
First sort predicted bounding boxes from high -> low confidence.

Then for each detection in the sorted detections:
- If it matches some ground truth bounding box with [[IoU]] > 0.5, mark is as positive (correct) and eliminate the ground truth bounding box. Note that you can choose whatever classification criteria you want (this is a hyperparameter).
- If it doesn't match, mark it as negative
- Plot a point on the precision-recall curve.

### Example

![[sort-preds-by-conf.png|250]]
We first order the predicted bounding boxes from highest confidence to lowest confidence.

---
![[first-predicted-matches-gt.png|250]] ![[pr-first-point.png|250]]

For instance, if our first prediction correctly matches with a ground truth bounding box, we will mark it as positive and remove the ground truth bounding box. We will then plot this point on the precision-recall curve. It has precision 1/1 because out of all the predicted 
boxes considered so far, we were correct on 1/1 of them (how many of the predicted boxes were correct). It has recall 1/3 because we have only correctly predicted 1/3 of the ground truth bounding boxes (how many of the ground truth boxes have we predicted).

--- 

![[second-predicted-matches-gt.png|250]] ![[pr-second-point.png|250]]

We can repeat this process for the second prediction. It is true again. Our precision remains 1.0 and our recall increases.

--- 

![[third-pred-no-match.png|250]] ![[pr-third-point.png|250]]


We repeat the process again, but this time we were incorrect. Our precision decreases, but our recall stays the same.

---


![[fourth-no-match.png|250]] ![[pr-fourth-point.png|250]]

We then make another incorrect prediction

---
![[pr-last-pred.png|250]] ![[pr-last-point.png|250]]
We correctly predict the last bounding box

----

Our final precision is 3/5 because out of 5 of our predicted bounding boxes, 3 were correct. Our final recall is 3/3 because we correctly predicted all of the bounding boxes.