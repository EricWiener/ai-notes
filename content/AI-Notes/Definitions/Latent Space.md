---
tags:
  - flashcards
source: 
summary: 
aliases:
  - Latent Variable Models
---
In the context of AI, latent space refers to a lower-dimensional representation of data that captures meaningful features and patterns. It is a mathematical abstraction or space where complex data is projected or encoded into a more compressed and structured form. This is achieved through techniques like autoencoders, variational autoencoders (VAEs), and generative adversarial networks (GANs).

During training, the AI model tries to capture the underlying structure of the data and create a compact representation that retains the most important information while discarding irrelevant details. Once the latent space is learned, it can be used for various tasks such as data generation, data compression, data exploration, and even recombining existing data to generate new samples.

The latent space enables AI models to perform tasks like image generation, style transfer, anomaly detection, and data visualization. By manipulating points in the latent space, it is possible to control and generate new data samples that share similar characteristics to the original dataset, allowing for creative applications in AI.

# Latent Variable Models
A latent variable model (LVM) $p$ is a probability distribution over two sets of variables $x, z$:
$$p(x,z;\theta)$$
where $x$ variables are observed at training time in a dataset $D$ and the $z$ variables are never observed.

For example, you could have a [[Generative Adversarial Networks|GANs]] where the generator predicts an image from a noise distribution and then the discriminator says where the predicted image is real or not.
![[AI-Notes/Generative/generative-adversarial-networks-srcs/Screen_Shot 3.png]]
At training time you can see the real images $x$, but you don't directly observe the noise distribution $z$ and can just sample points from it.