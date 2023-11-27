---
tags: [flashcards, eecs498-dl4cv]
source:
summary: train a student model to match predictions from a teacher model
---
![[distillation-diagram.png]]
With distillation you first train a teacher model on examples and ground-truth labels (using a loss function like [[Cross Entropy Loss]]). You then train a student model to match the prediction distribution of the teacher model (vs. trying to match the exact label of the example). You can use a loss like [[KL Divergence Loss]] to compare the predictions of the student and the teacher. You could also do something like using the argmax of the teachers predictions and using this as the ground truth for the student.

The idea is that the teacher model might be very good and therefore the probability distribution it predicts will give the student model more information than just training with one-hot labels.

You often use distillation when you want to train a very small model. The teacher model is a very large model and the studet model is very small. If you train a small model from scratch on a dataset, it tends to perform worse than if you trained the small model to match the predictions of a larger teacher model. Also, it is possible for the student model to get better performance than the teacher model.

![[second-loss-term-distillation.png]]
You can also sometimes add a second loss term to the student model to predict the ground truth label.

Note: distillation is not really a form of regularization since the student model is very small, so it is hard for it to overfit.

### Semi-supervised learning via distillation
You can train the teacher model on a dataset with you have ground-truths for your examples. Then, you can train your smaller model on a dataset where you don't have the ground truth and you can just use the predictions of the larger model.