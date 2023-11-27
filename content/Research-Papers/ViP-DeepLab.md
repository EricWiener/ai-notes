---
tags: [flashcards]
source: 
summary:
---

[Papers with Code](https://paperswithcode.com/method/vip-deeplab)

**ViP-DeepLab** is a model for depth-aware video panoptic segmentation. It extends [[Panoptic-DeepLab]] by adding a depth prediction head to perform monocular depth estimation and a next-frame instance branch which regresses to the object centers in frame t for frame t+1. This allows the model to jointly perform video panoptic segmentation and monocular depth estimation.