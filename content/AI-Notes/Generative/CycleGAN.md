---
tags: [flashcards, eecs442]
source:
summary: GAN model used to handle unpaired data.
---

The CycleGAN is an extension of the GAN architecture that involves the simultaneous training of two generator models and two discriminator models.

One generator takes images from the first domain as input and outputs images for the second domain, and the other generator takes images from the second domain as input and generates images for the first domain. Discriminator models are then used to determine how plausible the generated images are and update the generator models accordingly.

This extension alone might be enough to generate plausible images in each domain, but not sufficient to generate translations of the input images.

![[AI-Notes/Generative/generative-adversarial-networks-srcs/Screen_Shot 13.png]]

Most of the time in ML you work with paired data. For instance, you take a picture of a cat and then detect edges. You can use the edges and the final picture to train a model to convert from sketches to cats.

However, if you want to convert horses to zebras, you don't necessarily have pairs of images. Now we don't have pairs of inputs and outputs.

We could go back to our original formulation where we didn't pass the input image to the discriminator, but now we have nothing to force the output to correspond to the input.

![[AI-Notes/Generative/generative-adversarial-networks-srcs/Screen_Shot 14.png]]

We can use **CycleGAN** to handle this. 

### CycleGAN
The CycleGAN uses an additional extension to the architecture called cycle consistency. This is the idea that an image output by the first generator could be used as input to the second generator and the output of the second generator should match the original image. The reverse is also true: that an output from the second generator can be fed as input to the first generator and the result should match the input to the second generator.

Cycle consistency is a concept from machine translation where a phrase translated from English to French should translate from French back to English and be identical to the original phrase. The reverse process should also be true.

You want the return translation to English to be as close to the original version as possible. We want going from horses → zebra → back, we want to get a similar image.

Given an image $X$, we will map it to $Y$ (horse to zebra). We will then apply an inverse mapping $F$ (zebra to horse) and go from $Y$ to $X$ and get a similar result to the original image.

![[AI-Notes/Generative/generative-adversarial-networks-srcs/Screen_Shot 16.png]]

We want to be able to go horse → zebra → horse and also zebra → horse → zebra. We use both losses $||F(G(x)) - x||_1$ and $||G(F(y)) - y||_1$. You end up learning a model that can convert back and forth between the two.

You can also use this to train on a large dataset of a certain artist and then modify an input image. You no longer need just a single painting to transfer the style, but can train on a large amount of the artist's work.

![[AI-Notes/Generative/generative-adversarial-networks-srcs/Screen_Shot 17.png]]
