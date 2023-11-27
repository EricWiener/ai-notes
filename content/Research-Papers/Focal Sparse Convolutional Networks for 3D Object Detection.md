---
tags: [flashcards]
aliases: [Focal Sparse Convolution]
source: https://arxiv.org/abs/2204.12463
summary: introduce Focal Convolutions which modify sparse convolutions ot only apply feature dilations at important positions (dynamically predicted).
---
[YouTube Video](https://youtu.be/xY5gS9g5C6c)


![[image-20230711151009995.png]]
Figure 1:Process of different sparse convolution types. Submanifold sparse convolution fixes the output position identical to input. It maintains efficiency but disables information flow between disconnected features. Regular sparse convolution dilates all input features to its kernel-size neighbors. It encourages information communication with expensive computation, as it seriously increases feature density. The proposed focal sparse convolution dynamically determines which input features deserve dilation and dynamic output shapes, using predicted cubic importance. Input and Output are illustrated in 2D features for simplification. This figure is best viewed in color.

### Benefits
- It has a better receptive field and information flow than [[Submanifold Sparse Convolutional Networks|Submanifold Sparse Convolution]].
- It has lower computational cost than [[Submanifold Sparse Convolutional Networks|Sparse Convolutions]].
- It has a meaningful interpretation and you can visualize the important positions in each scene.