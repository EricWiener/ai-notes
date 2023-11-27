---
tags: [flashcards]
source: https://paperswithcode.com/method/masked-convolution
tldr: convolution that masks certain pixels so the model can only predict based on pixels it has seen (used for generative models).
---
A **Masked Convolution** is a type of [convolution](https://paperswithcode.com/method/convolution) which masks certain pixels so that the model can only predict ==based on pixels already seen==. This type of convolution was introduced with [PixelRNN](https://paperswithcode.com/method/pixelrnn) generative models, where an image is generated pixel by pixel, to ensure that the model was conditional only on pixels already visited.
![[masked-convolution-20220227092100827.png]]
<!--SR:!2024-09-19,697,310-->

