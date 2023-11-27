---
tldr: Learn visual features on unlabeled images with Pretext Tasks
tags: [flashcards]
---

Learn visual features on ==unlabeled images== with ==[[Pretext Tasks]]==
<!--SR:!2027-01-14,1314,310!2024-04-20,306,250-->

![[example-of-self-supervised-learning.png]]
- Ex: start with unlabelled images, perform random image augmentations on these images, extract features from the augmented patches, and then you want the patches from the same source image should give similar features. The learning signal is that patches from the same source image should have similar features after passing through the network.
- Strong empirical results. Unsupervised pretraining on ImageNet images performs similarly to using supervised labels.

### Problems
- **No ==[[semantics]]==**. No learning signal to unify features of visually dissimilar instances (ex. two patches that contain a dogs head will be treated as dissimilar since they come from different images - however, they should be treated as similar features).
- **Learning from unlabeled images may be ==too hard==**. Images usually come with some metadata, so we are making this problem harder than it needs to be. We shouldn't try to force our models to learn images without any extra information.
<!--SR:!2026-02-28,1036,290!2026-12-14,1364,350-->