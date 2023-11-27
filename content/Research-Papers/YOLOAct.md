---
tags: [flashcards]
source:
summary: extension of YOLO that predicts masks.
---

This is the one-stage counterpart to [[Mask R-CNN]].

You generate $k$ prototype masks ($k$ is a hyperparameter that is unrelated to the number of classes). This is similar to the number of anchors that you use in [[YOLO]].

![[yoloact-mask-co-efficients.png]]
For each anchor you end up with $k$ coefficients (one per mask prototype) that tell you how much each mask corresponds to each anchor. You then compute a weighted combination of each mask (using the coefficients as weights) to create the final prototypes.

![[yoloact-mask-assembly.png]]
The weights for each mask can be positive or negative (ex. negative if this is the mask for another object). In the above example you have some masks for the person and you subtract the mask for the racket to segment just the person.

YOLOAct is faster than [[Mask R-CNN]] and has better mask predictions for larger objects, but worse predictions for smaller objects.

