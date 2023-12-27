---
tags:
  - flashcards
source: https://arxiv.org/abs/1904.07850
summary: this paper uses a fully convolutional network to predict a centerpoint heatmap to locate where object centers are
publish: true
---

The paper represents objects by a single point at their bounding box center. Other properties, such as object size, dimension, 3D extent, orientation, and pose are then regressed directly from image features at the center location. Object detection is then a standard keypoint estimation problem. They feed the input image to a fully convolutional network that generates a heatmap. Peaks in this heatmap correspond to object centers. Image features at each peak predict the objects bounding box height and weight.

![[screenshot 2023-07-13_13_07_08@2x.png]]
> We model an object as the center point of its bounding box. The bounding box size and other object properties are inferred from the keypoint feature at the center. Best viewed in color.