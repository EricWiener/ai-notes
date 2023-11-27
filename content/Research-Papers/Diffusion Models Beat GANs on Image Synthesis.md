---
tags: [flashcards]
source: https://arxiv.org/abs/2105.05233
aliases: [Classifier Guidance]
summary: paper from OpenAI that is a follow up to [[Denoising Diffusion Probabilistic Models]]. They use additional normalization, residual connections, etc.
---

[YouTube Paper and Code Overview](https://youtu.be/hAp7Lk7W4QQ)

# Summary
- They improve their U-Net Architecture from the [[Denoising Diffusion Probabilistic Models]] paper.
- The previous paper conditioned models by including class information via normalization layers. This paper introduces using classifier guidance to additionally condition models.
- Using a gradient scaling term they are able to trade off between higher quality samples and more diverse samples.

The paper was motivated to close the gap between diffusion models and GANs. They noticed the following reasons why GANs outperform diffusion models:
- GANs have been studied heavily and the architectures have been heavily refined.
- GANs are able to trade off diversity for fidelity, producing high quality samples but not covering the whole distribution (see StyleGAN for how this is done).

They experimented with various architecture and settled on the following:
> variable width with 2 residual blocks per resolution, multiple heads with 64 channels per head, attention at 32, 16 and 8 resolutions, BigGAN residual blocks for up and downsampling, and adaptive group normalization for injecting timestep and class embeddings into residual blocks.

# Classifier Guidance
The previous paper, [[Denoising Diffusion Probabilistic Models]], incorporates class information using normalization layers which is how timestep information is included as well.

An alternative way is to use classifier guidance and train a model $p(y|x)$ to improve a diffusion generator. This is using a traditional discriminator model that takes an input $x$ and predicts the label $y$.

Existing approaches try conditioning a pre-trained diffusion model on the gradients of a classifier. You can train a classifier $p_\phi\left(y \mid x_t, t\right)$ on noisy images $x_t$ and then use ==gradients== $\nabla_{x_t} \log p_\phi\left(y \mid x_t, t\right)$ to guide the sampling process towards an arbitrary class label $y$.
<!--SR:!2024-06-18,261,310-->

The gradient tells you how you need to tweak the current image to maximize the class label of your image. Usually when training a model you take the gradient with respect to your model weights, but now we are taking the gradient with respect to the input. This then tells you how to change your image to maximize the likelihood the image belongs to a particular class.

### Conditional Reverse Noising Process
A regular diffusion model without any label conditioning can be denoted $p_\theta\left(x_t \mid x_{t+1}\right)$. To condition this on a label $y$, you can sample each transition (going from noisy image at $t + 1$ to $t$) using:
$$p_{\theta, \phi}\left(x_t \mid x_{t+1}, y\right)=Z p_\theta\left(x_t \mid x_{t+1}\right) p_\phi\left(y \mid x_t\right)$$
where $Z$ is a normalizing constant. You typically can't sample from this distribution, but it's possible to approximate sampling from it using:
$$\log(p_{\theta, \phi}\left(x_t \mid x_{t+1}, y\right)) \approx \log p(z)+C_4$$
where $C_4$ is a constant and $z \sim \mathcal{N}(\mu+\Sigma g, \Sigma)$ is a value sampled from a Gaussian distribution with its mean shifted by $\Sigma g$ where $g=\left.\nabla_{x_t} \log p_\phi\left(y \mid x_t\right)\right|_{x_t=\mu}$ is the gradient with respect to the input image for the classifiers prediction for label $y$. [See math here](https://youtu.be/hAp7Lk7W4QQ?t=646). Note that the $\log$ is applied to help simplify the expression. Note that $\Sigma$ is the variance and in this version of diffusion model we predict both the mean $\mu$ and variance $\Sigma$.

> [!NOTE]
> If we want to sample from the complex and intractable distribution $Z p_\theta\left(x_t \mid x_{t+1}\right) p_\phi\left(y \mid x_t\right)$ all we need to do is sample from a Gaussian with a shifted mean $\mathcal{N}(\mu+\Sigma g, \Sigma)$ where $g$ is the gradient of our classifier.
> 
> This is what the paper refers to when it says: "We have thus found that the conditional transition operator can be approximated by a Gaussian similar to the unconditional transition operator, but with its mean shifted by $\Sigma g$."

### Sampling new images
The following algorithm shows how to sample new images conditioned on a particular label starting from random noise.
![[sampling-new-images.png]]

**For a unconditioned diffusion model:**
- You take a noisier image $x_t$ and then predict the mean and variance for the slightly de-noised version. 
- You then sample $x_{t-1}$ from this.

**For a conditioned diffusion model:**
- You first pass the noisier image $x_t$ to a classifier and get class predictions for it. You then calculate the gradient of the input image with respect to the correct label $y$. This is $g$.
- You then pass $x_t$ to a diffusion model that predicts the mean and variance for the slightly de-noised version.
- You then shift the mean by the gradient $g$ and a weighting term $s$.
- You then sample $x_{t-1}$ from the shifted distribution.

![[conditioned-and-unconditioned-diffusion-model.excalidraw|900]]

**Intuition for what shifting the mean does:**
$g=\left.\nabla_{x_t} \log p_\phi\left(y \mid x_t\right)\right|_{x_t=\mu}$ tells you how to tweak the image to maximize the confidence for a certain class $y$. You start with your initial distribution $\mathcal{N}(\mu, \Sigma)$ and then shift it to $\mathcal{N}(\mu+\Sigma g, \Sigma)$. Below shows a 3D visualization of what this looks like. You have your initial distribution prediction and then you shift its center by the vector $\Sigma g$.
![[diffusion-model-shifting-distribution.excalidraw]]

### Scaling Classifier Gradients
They scale the classifier gradient $\Sigma g$ by a scalar $s > 1$ which they saw give better results empirically. The gradient scaling is done with $s \cdot g$ which is equivalent to $s \cdot \nabla_x \log p(y \mid x)$. Using rules of logs ($\log _b\left(M^k\right)=k \cdot \log _b M$), we can pull the exponent $s$ inside:
$$s \cdot \nabla_x \log p(y \mid x)=\nabla_x \log [\frac{1}{Z} p(y \mid x)^s]$$
where $Z$ is a constant that can be dropped since we are taking the gradient with respect to $x$. When $s > 1$, the distribution $p(y|x)$ becomes sharper (lower spread and higher peak) as can be seen in the following plot which shows $N(0, 1)^s$ to varying powers of $s$.

![[diffusion-models-beat-gans-on-image-synthesis-20230622151018620.png]]
[Generated with this colab notebook](https://colab.research.google.com/drive/16EZhQFAFaL9iXS5OcddPcHJA7-F9XVIU?usp=sharing)

Reducing the spread of the distribution via a larger gradient scale focuses more on the modes of the classifier (most frequent values) which is desirable for producing ==higher quality (but less diverse)== samples.
<!--SR:!2024-06-20,263,310-->

> [!NOTE] Using a larger gradient scale improves performance
> Using a high enough gradient scale, the guided unconditional model can get close to the [[FID Score]] of an unguided conditional model (which includes label information via the normalization layers). Using a conditional model with guiding yields the best FID.


### Training the classifier
The classifier model is the downsampling trunk of a UNet model with an attention pool at the 8x8 layer. The classifiers are trained on ImageNet where the images are noised with the same noising distribution used for the inputs to the corresponding diffusion model. The pipeline looks like (ImageNet image, GT label) -> (noised image, GT label) -> predict on noised image -> optimized based on loss of predictions vs. GT label. They also add random cropping to reduce overfitting of the classifier.

> [!NOTE] You need to train the classifiers on noised images
> This is because at inference time you will pass a noised image from time $t=T$ to $t=1$ and iteratively de-noise it. You need the classifier to see similar distributions of images during training so it produces accurate label predictions.

Note that this type of approach is limited to training the classifier on labeled datasets. You can extend this approach to condition diffusion models using other types of models such as conditioning an image generator with a text caption using a noisy version of [[CLIP Learning Transferable Visual Models From Natural Language Supervision|CLIP]].