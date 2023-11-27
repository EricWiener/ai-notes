---
tags: [flashcards]
source: https://gregorygundersen.com/blog/2018/04/29/reparameterization/
summary: the reparameterization trick is a way to let you sample from a normal distribution. It is useful for VAE autoencoders and Diffusion Models.
---
The reparameterization trick is a way to let you sample from a normal distribution. If you have a normal distribution represented by $\mathcal{N}\left(\mu, \sigma^2\right)$, you can draw a sample from it using $\mathcal{N}\left(\mu, \sigma^2\right)=\mu+\sigma \cdot \epsilon$ where $\epsilon$ is some white noise generated from a normal distribution with mean 0 and standard deviation 1 ($\mathcal{N}(0, \mathbf{I}))$.