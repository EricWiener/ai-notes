---
tags: [flashcards]
aliases: [RS-Lane]
source: https://www.hindawi.com/journals/jat/2021/7544355/
summary: simultaenously performs semantic and instance segmentation to detect an unlimited number of lanes using per-pixel embeddings.
---

# Introduction
There are two main approaches to lane detection:
- Traditional CV approach which uses features like color, edges, or geometric features. These methods are simple and efficient, but they need to manually adjust the parameters. Although they can perform well when working in normal situations, they cannot adapt to situations with different conditions such as lighting and occlusion.
- The other approaches are deep learning based and typically use convolutional networks.

### Difficulties with lane detection
- You need to detect a dynamic number of lanes if segmenting individual lanes.
- The number of background pixels is far greater than the number of lane pixels so learning to predict lane markings can be difficult. Side note: you could try to address this using weights with [[Cross Entropy Loss#Binary Cross-Entropy Loss (aka Log Loss)|BCELoss]], [[F1 Score|Dice Loss]], or [[Focal Loss]].
- Some situations are difficult because you have no lines, shadow occulusion, vehicles over the lines, and complex lighting conditions.
- You have additional complications of sometimes the lane boundary might be a wall, guard rail, or median.
- Lines can take on many forms: yellow lines, white lines, dashed lines, solid lines, bumps, reflectors, etc.
- Lines look different at different speeds.
- Lidars can look at reflectance but lidars are expensive and lines could be destroyed.

### Pixel Embeddings
The paper uses pixel embeddings to achieve instance segmentation without a limit on the number of lanes that can be detected. The pixel embeddings are based on [[Semantic Instance Segmentation with a Discriminative Loss Function]].

### Self-Attention Distillation (SAD)
[Source](https://arxiv.org/abs/1908.00821)
This is a training procedure and doesn't change inference speeds. It improves the network without needing more annotation information. This was likely used because the TuSimple dataset they used wasn't very large.

allows a model to learn from itself without any additional supervision or labels

### Split Attention
[Source](https://arxiv.org/abs/2004.08955)
This was used to improve the feature representation of the network on slender and sparse annotations like lane markings.

# Pipeline
### Preprocessing
- They downsample the image using bilinear interpolation.
- They have two labels per pixel since they have two branches (one for instance and one for semantic segmentation). The first label is the binary branch and says whether a pixel belongs to lanes or background. The other label is for the embedding branch which denotes which lane the pixel belongs to.

### Model training
