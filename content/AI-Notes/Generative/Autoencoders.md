---
tags: [flashcards, eecs442, eecs498-dl4cv]
aliases: [VAE, VAEs]
source: https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf
summary:
---

Relevant files: lec16-representation.pdf, 498_FA2019_lecture19.pdf, 498_FA2019_lecture20.pdf

# Autoencoders (regular, non-variational)
![[AI-Notes/Generative/autoencoders-srcs/Screen_Shot.png]]
An autoencoder is **an unsupervised learning method for learning feature representations from raw data ($x$) without any labels**. You train the model by trying to get it to copy its input to its output while passing through a lower-dimensional feature space. For instance, given a set of images, return a set of reconstructed images using their most representative features. **A traditional autoencoder is not a probabilistic model.**

**Algo:**
- Input your original images, each with $d$ dimensions
- Compress your images down to $k$ dimensions where $k < d$ (encode)
- Reconstruct your images back to $d$ dimensions (decode)

> [!NOTE] You want to compress (encode) and decompress (decode) the images such that the reconstructed images are close to original image. During training, you want to penalize inaccurate reconstructions.

**Encoder**: $h = f(x)$. $h$ is the encoded image.

**Decoder**: $r = g(h)$. $r$ is the re-constructed image.

**Optimize:** $f, g = \text{arg min}_{f, g}\|X-g(f(x))\|^2$. This is finding the encoder and decoder function that minimize the L2 loss from the reconstructed data and the original data.

Want to get $g(f(h))$ to be as close to $x$ as possible. We evaluate loss in terms of $L(x, g(f(x))$ because we want the output of the autoencoder to be as close to the original as possible. **The perfect autoencoder would just be the identity function.** You can use whatever architecture you want for the encoder and decoder (neural networks, CNN, etc.).

**Layers:**
- Convolutional: shrinks input. These are in the encoder.
- Deconvolutional: grows input. These are in the decoder.

**Purpose:**
- To learn most representative features of image
- To get a lower dimensional feature representation
- You don't care about the decoder after training
- The weights from an autoencoder can be used to initialize a supervised model. You can then train for the final task (sometimes with small amounts of labelled data).

> [!note]
> In practice, autoencoders aren't used much. They aren't a very practical approach to unsupervised feature learning and don't perform so well. They were used in the early/mid 2000s, but now people just train networks from scratch.
> 

**Forms:**
- Undercomplete network: learn a dimension smaller than the original dimensionality
    - This captures the most important features of the training data
- Overcomplete network: learn a dimension larger than the original dimensionality

**Similarity to [[Principal Component Analysis|PCA]]**: If the encoding and decoder functions were linear and the loss was mean square error, the autoencoder would function the same as principal component analysis.

### Regularization
Need to limit the flexibility of the model to encourage the model to do something other than just copy the data. You want to enforce sparsity of representation to be robust to noise.

### De-noising autoencoder
We want to be able to take noisy data and generate clean data. This is difficult because it’s hard to distinguish between noise and signal. Autoencoders use mean squared error to compare the output to the original image.

- You could blur the input image in order to get a prediction target for your output.
- You could also add noise to the inputs and then train the model to recreate the original data.

# Variational Autoencoders
### VAE TLDR
Variational Autoencoders have an encoder that produces a distribution over a latent $z$ given a data input $x$. You then have a decoder that takes the distribution as input and produces data $\tilde{x}$ by producing a distribution over the data space which you can then sample from.

![[vae-overview-diagram.png]]

### Overview
With [[Autoregressive Models|PixelRNN]] and [[Autoregressive Models|PixelCNN]] we explicitly parameterized the density function with a neural network. We could get the likelihood of a specific image (you can go through all your training data and calculate the likelihood of a certain pixel given the proceeding pixels) and we trained the model to maximize the likelihood of the training data: $p_{\theta}(x)= \Pi_{t=1}^Tp(x_t|x_1, ..., x_{t-1})$ (i.e. learn to predict the most likely color for the next pixel $x_t$ given all the preceeding pixels). With Variational Autoencoders (VAE), we can no longer get the likelihood of a specific image. We will instead use an **intractable density** that we cannot explicitly compute or optimize. We can, however, get a **lower bound on the density function** (lots of complicated math involved in creating the equation that gives the lower bound) , so we will try to maximize this lower bound on the density.

Variational autoencoders have an improvement over [[Autoregressive Models]] because they can learn features about the images. Autoregressive models just use the raw pixels to try to predict the next pixel. Variational autoencoders have a deeper understanding of the images.

### Variational Autoencoders vs. Regular Autoencoders
With regular autoencoders, we couldn't generate a new image. You would learn an encoder that takes an input and produces a lower-dimensional version. The corresponding decoder learns to take the lower-dimensional features and then generate something close to the original input to the encoder. However, there are regions of the latent feature space (input to the decoder) that don't correspond to any encoder output and if you gave that input to the decoder you will just get garbage output. Once the network is trained, and the training data is removed, we have no way of knowing if the output generated by the decoder from a randomly sampled latent vector is valid or not. Hence regular autoencoders are mainly used for compression or for pre-training.

Traditional autoencoders are trained to reconstruct an input after compressing it. Variational autoencoders are trained to reconstruct an input after compressing it **and** to make sure that the latent space is ==regularized==. The additional constraint makes it so you can sample/generate new images since all points in the latent space are legit inputs to the decoder (you won't just get garbage out).
<!--SR:!2024-06-19,262,310-->

Just like traditional autoencoders, the input is passed through a series of layers (parameterized by the variable $\theta$) reducing its dimensions to achieve a compressed latent vector **_z_**. However, the latent vector is not the output of the encoder. **Instead, the encoder outputs the mean and the standard deviation for each latent variable. The latent vector is then sampled from this mean and standard deviation which is then fed to the decoder to reconstruct the input**. The decoder in the VAE works similarly to the one in traditional autoencoder.
[Source](https://towardsdatascience.com/difference-between-autoencoder-ae-and-variational-autoencoder-vae-ed7be1c038f2).

### Variational Autoencoders Problem Setup
Variational encoders are a probabilistic upgrade on regular autoencoders where we can **now generate new images**. You want to:

1. Learn latent features $z$ from raw data
> [!NOTE] Latent variables (as opposed to observable variables) are variables that are not directly observed, but inferred through a model from the observed variables.

2. Sample from the model to generate new data

We assume we have a training dataset with a bunch of unlabelled samples. The training data $\{x^{(i)}\}^N_{i=1}$ is generated from an unobserved (latent) representation $z$. 

> [!note]
> $x$ is an image, $z$ is latent factors used to generate $x$: attributes, orientation, objects in image, etc. You aren't able to observe $z$ for any $x$'s and need to learn it instead.
> 

### Sampling new data
During training, you learn a set of features $z$ that represent the data. You then sample from the feature space $z$ and try to generate a new image that maximizes the likelihood of that image conditioned on the features you selected $p(x|z^{(i)})$. 

We usually assume a simple distribution of the prior $p(z)$ (e.g. a Gaussian with a diagonal covariance, mean of zero, and unit variance) when sampling from the feature space. $p(z)$ is typically not learned and is a fixed distribution.

When we sample from the feature space and use a Gaussian with a diagonal covariance, this means that all the pixels are independent of each other (conditioned on z). You end up generating a mean and variance for every pixel in the image individually, which gives you a distribution over all possible images. For each pixel you can take the most likely value and this will allow you to generate an image.

### Training
If you could observe $z$ directly then you could train a [[Discriminative vs. Generative Models#**Conditional Generative Model:**|Conditional Generative Model]] to maximize the likelihood of an image $x^{(i)}$ given $z^{(i)}$. However, you can't directly observe $z^{(i)}$ so this is more difficult.

During training, you want to **maximize the likelihood of the data** given the model you learn. 
![[AI-Notes/Generative/autoencoders-srcs/Screen_Shot 1.png]]
The lower-bound on the log of the data likelihood is shown in the top equation

The [lecture](https://www.youtube.com/watch?v=Q3HU2vEhD5Y&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) covered a lot of specific math details about variational autoencoders, but the main takeaways are that:

- We are trying to learn a probabilistic distribution of the features
- We want to be able to generate new data
- We aren't able to calculate all the probabilities exactly, so we introduce another neural network to approximate one of the probabilities (this is what makes it **variational**). **Variational inference** is when you introduce another network to compute an ==intractable [[Posterior Distribution|posterior distribution]]==.
- We train the encoder and decoder networks together at the same time to maximize the lower-bound of the log-likelihood of the data.
- The images they generate aren't fantastic, but the math is nice (to someone at least).
- You can't compute a true probability of the data because some of the values are intractable. Instead, you can compute a lower-bound for them.
<!--SR:!2024-02-29,95,212-->

### Math


**What makes the posterior distribution intractable?**
The posterior probability is the probability of event A occurring given that event B has occurred $P(A|B)$. It can be calculated with [[Bayes' Theorem]].

![[screenshot-2023-04-06_19-30-07.png]]

For autoencoders we want to maximize the probability that a model with parameters $\theta=\left(\theta_1, \ldots, \theta_d\right)$ generated the training data $x=\left(x_1, x_2, \ldots, x_n\right)$ based on some latent variables $z$. This can be expressed as the following:
$$p_\theta(x)=\frac{p_\theta(x \mid z) p_\theta(z)}{p_\theta(z \mid x)}$$
- $p_\theta(x \mid z)$ is the probability of generating $x$ given the latent variables $z$. This can be calculated via the decoder network which outputs a mean and variance for a certain latent feature $z$.
- $p_\theta(z)$ is the probability of the specific latent variable $z$
- ${p_\theta(z \mid x)}$ is the probability that we get the specific latent variable $z$ given $x$. We want to approximate this using the encoder which will predict a distribution over $z$ given an input $x$. We denote the encoder as $q_\phi(z \mid x)$ and we want to ensure that $q_\phi(z \mid x) \approx p_\theta(z \mid x)$.


For autoencoders, we want to maximize the probability of a model $p(x \mid \theta)$ with parameters $\theta=\left(\theta_1, \ldots, \theta_d\right)$ given the training data $x=\left(x_1, x_2, \ldots, x_n\right)$. This model predicts $x$ with its parameters $\theta$. The posterior distribution of the Bayesian model (how likely are the models parameters given the training data) can be calculated with:
$$p(\theta \mid x)=\frac{p(x \mid \theta) p(\theta)}{\int p(x \mid \theta) p(\theta) d \theta}$$



https://datascience.stackexchange.com/questions/80272/what-makes-the-posterior-intractable

https://lips.cs.princeton.edu/variational-inference-part-1/

@TODO(ewiener): finish this by reading /Users/ewiener/Library/CloudStorage/GoogleDrive-ecwiener@umich.edu/My Drive/UMich/Old Classes/EECS 498 - DL4CV/Slides/L19 - Generative Models (1).pdf

### Variational Autoencoders: Image Editing
![[AI-Notes/Generative/autoencoders-srcs/Screen_Shot 2.png]]
Editing an image by modifying its feature representation

You can encode the image and then modify its feature representation. When you pass the modified features to the decoder, you will get a modified image. If you are able to identify what features belong to what aspects of the photo, you can edit the photo, as shown above.

### Variational Autoencoders: Summary
Variational Autoencoders are the probabilistic spin to traditional autoencoders. They allow us to generate data. They have an intractable density function, so to train we need to derive and optimize a (variational) lower bound.

**Pros:**
- Principled approach to generate models
- Allows inference of $q(z|x)$, which can be a useful feature representation for other tasks
- Faster to generate images than PixelRNN/PixelCNN

**Cons:** 
- Maximizes lower bound of likelihood: okay, but not as good an evaluation as PixelRNN/PixelCNN
- The samples generated are blurrier and lower quality compared to GANs. This is likely because we make a diagonal Gaussian assumption.