---
tags: [flashcards]
aliases: [Contrastive Loss]
---
[Good Medium Article](https://medium.com/@maksym.bekuzarov/losses-explained-contrastive-loss-f8f57fe32246)

> [!note] You want to put together pairs that are similar and spread apart pairs that are dissimilar
> 
- Contrastive loss was first introduced in 2005 by Yann Le Cunn et al. in [**this paper**](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf) and its original application was in Dimensionality Reduction.
    - It penalizes “similar” samples for being ==far from each other== (ex. this could be calculated with Euclidian Distance).
    - “Dissimilar” samples are penalized for ==being to close to each other==, but in a somewhat different way — Contrastive Loss introduces the concept of “margin” — a ==minimal distance that dissimilar points need to keep==. So it penalizes dissimilar samples for being ==closer than the given margin==.
<!--SR:!2030-01-25,2252,350!2026-01-02,1075,332!2024-07-20,292,340!2024-07-14,287,341-->

- Dimensionality Reduction: Given a sample (a data point) — a $D$-dimensional vector, transform this sample into a  $d$-dimensional vector, where ==d ≪ D, while preserving as much information as possible==.
<!--SR:!2026-03-15,1076,310-->

- Contrastive loss can't just try to maximize distance between dissimilar points and minimize distance between similar points. This is because if you are pulling similar points together for one class, you are inevitably going to ==attract some dissimilar points== as well, which would ==increase the dissimilar-points loss==. Vice versa is also true — if you are pushing all the dissimilar points as far away as possible, you may also push away some similar points, which would increase the similar-points loss. In the iterative training process there comes some time where these two losses are stabilizing after some time, i.e. each new update is not moving the data points around. And this state is called the Equilibrium point.
![[screenshot-2022-02-23_14-22-48.png]]![[screenshot-2022-02-23_14-23-26.png]]
<!--SR:!2025-10-31,975,310!2027-08-31,1543,330-->

# EECS 442 Contrastive Learning
![[AI-Notes/Generative/contrastive-learning-srcs/Screen_Shot.png]]

When working with autoencoders, you don't really care about the part of the model that decodes the image, since you just want the encoded image features. You can use **contrastive learning** which just tries to learn features embeddings that are similar to similar images, but different from different images. 

- Images should have a high dot-product with themselves
- Images should have a lower dot-product with other images.
- You can also take random-crops (or other augmentations) of the input image and make train the model to say they are similar to the original image.

> [!note]
> You no longer need to perform the decoding step.
> 