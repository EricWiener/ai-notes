---
tags: [flashcards]
source:
summary:
---

We can organize the types of generative models using the following diagram:

![[types-of-generative-models-srcs/Screen_Shot.png]]

> [!QUESTION] Why are there two instances of Markov Chains?
> Markov Chains used to be very popular and there are different types that can be used in different contexts. A Boltzmann Machine approximates a density function while a GSN makes it easier to draw samples. These are not the same algorithms. Markov Chains is just a large family of algorithms, but it isn't commonly used nowadays.
 
### Explicit vs. Implicit Density
To break this down by level, we can first look at the first split:

![[AI-Notes/Generative/types-of-generative-models-srcs/Screen_Shot 1.png]]
A generative model assigns density functions to images. There are some generative models where after training, you can give it an image, and it will give you the likelihood of that image (explicit density). There are other models where you can't pull out a likelihood value, but you can sample from the distribution.

### Tractable vs. Approximate
![[types-of-generative-models-srcs/Screen_Shot 2.png]]
A tractable density function is one where you can input an image and receive the actual value of the density function. An approximate density function is one where you can't efficiently get the actual value of the density function, but you can approximate it. These approximations can be made with [[Autoencoders#Variational Autoencoders|Variational Autoencoders]] or **Markov Chains**.

### Markov Chain vs. Direct Sampling
![[screenshot-2022-11-09_09-21-44.png|400]]
With direct you can directly draw samples from $p(x$) vs. with Markov Chain you need to go through an iterative process in order to produce a sample. Sampling from a GAN is very straightforward.

### Autoregressive vs. Variational Autoencoders vs. GANs
![[types-of-generative-models-srcs/Screen_Shot 3.png]]

**Autoregressive Models:**
- Directly maximize p(data)
- Generate high quality images
- Slow to generate images
- No explicit latent codes (no feature embeddings)

**Variational Models:**
- Maximize lower-bound on p(data). Not p(data) directly
- Generated images are often blurry (pixels are generated independently from neighboring pixels. They just rely on the feature embedding).
- Very fast to generate images
- Learn rich latent codes (good feature embeddings)

**GANs:**
- Don't try to model p(data)
- Generate extremely high quality images
- Difficult to train because of dual losses
- Don't get a feature embedding for an image 

# Diffusion vs. GANs vs. VAEs
[Source](https://youtu.be/a4Yfz2FxXiY?list=PLPioWEh9FVPpXSr-UhTqrIjlLUKVB5tGK&t=51).
![[gan-ddm-vae.png|400]]
### [[Autoencoders|VAE]]
![[Diffusion models from scratch in PyTorch 1-5 screenshot.png]]
VAEs encode an image into a latent space. During training you then sample from the latent space and try to reconstruct the original image. During inference you can generate new images by sampling from the latent space and passing this to the decoder. You can sample quickly and get diverse samples, but the image quality is poor.

VAEs and Normalizing Flows produce diverse samples quickly but usually the quality is worse than GANs. They are usually easy to train.

### [[Generative Adversarial Networks]]
![[Diffusion models from scratch in PyTorch 1-30 screenshot.png]]
GANs will take noise and then generate a new image. The discriminator is trained to predict whether the image is fake or not. The generator is fast and you can get good results, but they are very difficult to train due to the adversarial setup which often causes vanishing gradients or mode collapse.

### [[Diffusion Models]]
![[Diffusion models from scratch in PyTorch 2-34 screenshot.png]]
Diffusion models will iteratively add noise during training and then use a model to remove the noise. During inference you start with noise and then iteratively remove the noise. The iterative reverse process makes inference slow.

While the encoder in GANs go from noise to an image in one shot, diffusion models break it into an iterative process so you repeatedly remove a bit of noise. This yields better results but takes longer.