---
tags: [flashcards]
source: https://www.youtube.com/watch?v=cS6JQpEY9cs
summary:
---

**Deep Generative Learning**: you have samples drawn from an unknown distribution. You then train a neural network. At inference time you can draw new samples from the neural network that hopefully mimic the training distribution.

![[diffusion-model-youtube-diffusion-model-summary.mp4]]

# Denoising Diffusion Models
Denoising diffusion models consists of two processes:
- Forward diffusion process that gradually adds noise to input. Eventually you end up with white noise.
- Reverse denoising process that learns to generate data by denoising. You start with white noise and learn to generate data.

![[tutorial-on-denoising-diffusion-based-generative-modeling-20230313102614340.png]]
### Forward Diffusion Process
![[tutorial-on-denoising-diffusion-based-generative-modeling-20230313102754380.png]]
At every step of adding noise you will use a normal distribution $q(x_t|x_{t-1})$ where $q$ takes $x_{t-1}$ ($x$ at the previous step) and generates $x_t$ ($x$ at the current step).

**[[Markov Chain|Markov Process]] to generate image one step at a time:**
$$q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right)=\mathcal{N}\left(\mathbf{x}_{\mathbf{t}} ; \sqrt{1-\beta_t} \mathbf{x}_{\mathbf{t}-\mathbf{1}}, \beta_t \mathbf{I}\right)$$
- $\mathcal{N}$: this is a normal distribution
- $\mathcal{N}(x_t;$ this is a normal distribution over the current step $x_t$.
- The mean is given by $\sqrt{1-\beta_t} \mathbf{x}_{\mathbf{t}-\mathbf{1}}$. $\beta_t$ is a small positive scalar value.
- The variance is given by $\beta_t \mathbf{I}$ (multiply the scalar by the identity matrix).
- At each step it will take the image from the previous step, $\mathbf{x}_{t-1}$, rescale it by $\sqrt{1-\beta_t}$ (very close to ~0.999) and adds a tiny amount of noise ($\beta_t \mathbf{I}$).

**Joint distribution:**
$$q\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right)=\prod_{t=1}^T q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right)$$
- This is the joint distribution for all the samples that will be generated from $\mathbf{x}_0$ to $\mathbf{x}_T$ starting from $\mathbf{x}_0$.
- The joint distribution is the product ($\prod$) of the samples formed at each step.

**Diffusion Kernel**:
You can define a diffusion kernel that will take you from time step $0$ to time step $t$ directly without having to apply the noise for each step in between.

We first define $\alpha_t = 1 - \beta_t$. We also define a new scalar $\bar{\alpha}_t=\prod_{s=1}^t\left(1-\beta_s\right)$ where $\bar{\alpha}_t$ is the product of $(1-\beta_s)$ for all steps from $s = 1$ to $s = t$ and the noise $\beta_s$ can change depending on the step number. 

The following is called the Diffusion Kernel and describes the distribution of possible noised images at step $t$:
$$\left.q\left(\mathbf{x}_t \mid \mathbf{x}_0\right)=\mathcal{N}\left(\mathbf{x}_t ; \sqrt{\bar{\alpha}_t} \mathbf{x}_0,\left(1-\bar{\alpha}_t\right) \mathbf{I}\right)\right)$$
The math behind how the Diffusion Kernel is created is explained [[Deriving the forward diffusion kernel|here]].

To sample noised images at time step $t$ using this distribution you can use the [reparameterization trick](https://gregorygundersen.com/blog/2018/04/29/reparameterization/): 
$$\mathbf{x}_t=\sqrt{\bar{\alpha}_t} \mathbf{x}_0+\sqrt{\left(1-\bar{\alpha}_t\right)} \epsilon \quad \text{ where } \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
- $\sqrt{\bar{\alpha}_t} \mathbf{x}_0$ is the mean
- $\sqrt{\left(1-\bar{\alpha}_t\right)}$ is the standard deviation
- $\epsilon$ is some white noise generated from a normal distribution with mean 0 and standard deviation 1.

**$\beta_t$ values schedule (noise schedule):**
The $\beta_t$ values change at each time step (you add a different amount of noise at each step). 
- $\beta$ follows a learning schedule and typically is a value $\beta_t \in (0, 1)$ and $\beta_0 < \beta_1 < \beta_2 < \ldots \beta_T$.
- $\sqrt{1 - \beta_t}$ will decrease over time as $\beta_t$ increases which will bring the mean closer to 0.
- $q(x_T | x_0)$ will approach $\mathcal{N}(0, 1)$.

You want the noise schedule designed so that $\bar{\alpha}_T$ (the $\bar{\alpha}$ value at the last timestep) approaches 0. This makes it so $\left.q\left(\mathbf{x}_t \mid \mathbf{x}_0\right)=\mathcal{N}\left(\mathbf{x}_t ; \sqrt{\bar{\alpha}_t} \mathbf{x}_0,\left(1-\bar{\alpha}_t\right) \mathbf{I}\right)\right)$ becomes  $\left.q\left(\mathbf{x}_T \mid \mathbf{x}_0\right) \approx \mathcal{N}\left(\mathbf{x}_T ; \mathbf{0}, \mathbf{I}\right)\right)$. This means your final noised image is approximately the standard normal distribution.

The paper chose a linear schedule for $\beta_t$ where $\beta_1 = 10^{-4} = 0.0001$ and $\beta_T = 0.02$ and $T = 1000$. This linearly increased over time so you add more noise over time. You want to use lots of small steps so that it is easier for the model to reverse the process. If you had a very small number of steps this would be more similar to a GAN which goes from noise to an image in one shot. The model would be faster at generating images but it would likely have worse results.

For large timestamps $\bar{\alpha}_t$ will be close to 0 since you are multiplying many decimals together so the number keeps getting smaller and smaller.

You rescale the images to have all RGB values lie between $[-1, 1]$. For instance, a red pixel (255, 0, 0) would be scaled to (1, -1, -1). The distribution of the red channel for this pixel for the noised image at time step $1$ would be described by $\mathcal{N}(\sqrt{1 - \beta_t} * 1, \beta_t \mathbf{I})$ or $\mathcal{N}(\sqrt{1 - \beta_t}, \beta_t \mathbf{I})$. A larger initial value of $\beta_t$ will shift the mean closer to 0 and the variance closer to 1 and therefore destroy the information in the image faster and move closer to the $\mathcal{N}(0, 1)$ distribution.

### What happens to distribution in forward diffusion
We have looked at the diffusion kernel $q\left(\mathbf{x}_t \mid \mathbf{x}_0\right)$ which tells us the distribution of possible $x_t$ that are produced for a certain $x_0$. We can look at the distribution of the diffused data $q(x_t)$ if we we marginalize over $x_0$ (sum over all the possible values of $x_0$) to make the distribution no longer conditional on a certain $x_0$.

![[tutorial-on-denoising-diffusion-based-generative-modeling-20230315091822210.png|500]]
- $q(x_t)$ is the diffused data distribution
- $q(x_0, x_t)$ is the joint distribution of the clean input image and the noised data
- $q(x_0)$ is the input data distribution
- $q(x_t|x_0)$ is the diffusion kernel that generates $x_t$ based on $x_0$. This is the distribution of all $x_t$ that can be produced for a given $x_0$. This is just a normal distribution.

> [!NOTE]
> The [[Computer Science/CS Definitions/Bayes' Theorem|Definition of Conditional Probability]] is that $$P(A \mid B)=\frac{P(B \mid A) P(A)}{P(B)}$$. This was used to go from $q\left(\mathbf{x}_0\right) q\left(\mathbf{x}_t \mid \mathbf{x}_0\right)$ to $q\left(\mathbf{x}_0, \mathbf{x}_t\right)$

If we just look at one dimensional data, this is what the distribution looks like. You can see at timestep $x_T$ that the data follows a normal distribution and is just noise.
![[tutorial-on-denoising-diffusion-based-generative-modeling-20230315092839854.png]]

The diffusion kernel is like applying a Gaussian convolution.

In practice we don't have access to the actual data distribution, but we can approximate it using [[Ancestral Sampling]] where you sample training data and then diffuse it to approximate sampling from the diffused data distribution.
- You first sample a clean training image $\mathbf{x}_0 \sim q\left(\mathbf{x}_0\right)$.
- You can then compute the forward diffusion process to generate $x_t$ based on $x_0$: $\mathbf{x}_t \sim q\left(\mathbf{x}_t \mid \mathbf{x}_0\right)$.
- This is how we approximate sampling $\mathbf{x}_t \sim q\left(\mathbf{x}_t\right)$.

### Generative Learning by Denoising
The diffusion process is designed so that you end up with a normal distribution at the last timestep: $\left.q\left(\mathbf{x}_T\right) \approx \mathcal{N}\left(\mathbf{x}_T ; \mathbf{0}, \mathbf{I}\right)\right)$.

**Mathematical interpretation of generated next de-noised step**:
![[tutorial-on-denoising-diffusion-based-generative-modeling-20230315094549979.png]]

To generate the original image from noise you first sample $x_T$ from a standard normal distribution $\mathbf{x}_T \sim \mathcal{N}\left(\mathbf{x}_T ; \mathbf{0}, \mathbf{I}\right)$ and then iteratively sample the less noised version $\mathbf{x}_{t-1} \sim q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)$. 

However, for most problems you don't have access to the true distribution $q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)$ because this is equivalent (via Baye's Rule) to $q\left(\mathbf{x}_{t-1}\right) q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right)$. You do know $q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right)$ (this is just a Gaussian distribution), but you don't know the distribution of all data at step $t-1$ ( aka $q\left(\mathbf{x}_{t-1}\right)$).

You can approximate $q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)$ using a normal distribution if $\beta_t$ at each small is very small in each forward diffusion step.

# Reverse Diffusion Process
**Using a model to predict the next de-noised step**
The forward diffusion process isn't learned. However, we train a model to do the reverse diffusion process. The reverse diffusion process is $p_\theta(\mathbf{x}_{t-1}|x_t)$ and tells us the distribution of $x_{t-1}$, a slightly less noisy value, given $x_t$, a noisier input. We use a model parameterized by $\theta$ to calculate this.

The distribution $p_\theta(\mathbf{x}_{t-1}|x_t)$ can be represented by the following distribution:
$$p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)=\mathcal{N}\left(\mathbf{x}_{t-1} ; \mu_\theta\left(\mathbf{x}_t, t\right), \sigma_t^2 \mathbf{I}\right)$$
where $\mu_\theta\left(\mathbf{x}_t, t\right)$ is a trainable network that takes in $x_t$ (the noised data) and $t$ (the time step of the noised data) and predicts the mean of the less noised version. $\mu_{\theta}$ just means that this network currently has parameters $\theta$. You then sample $x_{t-1}$ from the normal distribution with the mean predicted by the network and variance $\sigma_t^2 \mathbf{I}$.

You sample $x_T$ (most noised data) from the standard normal distribution $p\left(\mathbf{x}_T\right)=\mathcal{N}\left(\mathbf{x}_T ; \mathbf{0}, \mathbf{I}\right)$.

> [!NOTE]
> You use the learned model to predict the noise at each time step. You pass the time `t` in as an argument to the model.

### Noise Prediction Network
In order to calculate the distribution $p_\theta(\mathbf{x}_{t-1}|x_t)$ we need to know the mean $\mu_\theta\left(\mathbf{x}_t, t\right)$ and variance $\sigma_t^2 \mathbf{I}$. The variance just depends on $\alpha_t$ which is a fixed value, so we don't need to learn to predict this. Therefore, we just need to predict $\mu_\theta\left(\mathbf{x}_t, t\right)$. Using some fancy math we can see that predicting the noise $\epsilon_\theta$ is all that is needed to calculate $\mu_\theta\left(\mathbf{x}_t, t\right)$ since all other variables are known:
$$\mu_\theta\left(\mathbf{x}_t, t\right)=\frac{1}{\sqrt{1-\beta_t}}\left(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta\left(\mathbf{x}_t, t\right)\right)$$
- $\epsilon_\theta\left(\mathbf{x}_t, t\right)$ is a noise-prediction network that takes image $x_t$ and the time step $t$ and then predicts the noise used to generate $x_t$.
- You can then subtract this noise from $x_t$ and then re-scale and this gives you $\mu_\theta\left(\mathbf{x}_t, t\right)$.
- $\mu_\theta\left(\mathbf{x}_t, t\right)$ is the mean of the less noised version $x_{t-1}$ which you can then use as the mean of a distribution $p_\theta(\mathbf{x}_{t-1}|x_t)$ to sample $x_{t-1}$ from.
[See here for the math used](https://youtu.be/HoKDTa5jHvg?t=1320)

> [!NOTE] You predict the total noise $\epsilon$ used to go from $x_0$ to $x_t$ instead of directly predicting the mean of the less noised version $x_t$
> Calculating the noise is all you need to compute the distribution $p_\theta(\mathbf{x}_{t-1}|x_t)$ which you can then sample from to find $x_{t-1}$.

> [!NOTE] Noise prediction network inputs/outputs
> The input to the model is a noised image (`original_image + noise`) and the output of the model is the **total noise** (`noise`) that should be subtracted to go back to the original image (no noise).
> 

To actually sample $x_{t-1}$ given $x_t$ you can use the reparameterization trick and sample from $p_\theta(\mathbf{x}_{t-1}|x_t) = \mathcal{N}\left(x_{t-1} ; \frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta\left(x_t, t\right)\right), \beta_t\right)$ using:
$$x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta\left(x_t, t\right)\right)+\sqrt{\beta_t} \epsilon \quad \text{for }t >1$$
Note that when sampling for $t = 1$ for $p_\theta(\mathbf{x}_{0}|x_1)$ you don't add noise to the value (see [here](https://youtu.be/HoKDTa5jHvg?t=1590) for math why and [here](https://youtu.be/HoKDTa5jHvg?t=1650) for intuition):
$$x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta\left(x_t, t\right)\right)\quad \text{for }t =1$$
this is because when sampling $x_0$ from $x_1$ we want to predict the original noise-less image based on $x_1$. Our neural network will predict the noise $\epsilon_\theta\left(x_1, t\right)$ and we subtract this from $x_1$. We don't want to then add more noise via $\sqrt{\beta_t} \epsilon$ because we are already at the final de-noised image and the initial noise would just make the quality of $x_0$ worse.

![[diffusion-model-youtube-predict-noise.mp4]]

### Algorithms YouTube Overview
![[diffusion-model-youtube-algorithms.mp4]]

### Training
[YouTube Walkthrough](https://youtu.be/HoKDTa5jHvg?t=1620)
![[tutorial-on-denoising-diffusion-based-generative-modeling-20230324161819557.png|400]]
2. You sample a batch of samples from your training data
3. You sample a batch of timestamps from 1 to $T$
4. You sample random noise from a standard normal distribution.
5. You then take a gradient step based on the L2 loss between the noise used and the predicted noise used.
    - $\sqrt{\bar{\alpha}_t} \mathbf{x}_0+\sqrt{\left(1-\bar{\alpha}_t\right)}\epsilon = x_t$ uses the re-parameterization trick to predict $x_t$ based on $x_0$ and the noise $\epsilon$.
    - $\epsilon_{\theta}(\sqrt{\bar{\alpha}_t} \mathbf{x}_0+\sqrt{\left(1-\bar{\alpha}_t\right)}\epsilon, t) = \epsilon_{\theta}(x_t, t)$ will predict the noise used to generate $x_t$ based on the generated $x_t$ and the time step $t$ (note that $\epsilon_{\theta}$ doesn't have access to the $\epsilon$ noise that was used in the generation process or $x_0$).
    - You then calculate the L2 loss (take the squared magnitude of the difference between the actual noise used and the noise the noise-prediction network predicted was used).
6. Repeat process until converged.

Here is pseudo code for training the model:
```python
def train_loss(denoise_model, x_0, t):
    noise = torch.randn_like(x_0)

    x_noisy = q_sample(x_0=x_0, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    loss = F.l2_loss(noise, predicted_noise)
    return loss
```
Note that we predict the total noise added to the original image `x_0`. [Source for psuedo code](https://stats.stackexchange.com/a/601216).

### Sampling
[YouTube Walkthrough](https://youtu.be/HoKDTa5jHvg?t=1638)
To sample new images from your trained network you can use the following:
![[tutorial-on-denoising-diffusion-based-generative-modeling-20230324161904828.png|400]]
1. You start from the last step (the noisiest data) at $x_T$ and generate samples from a normal distribution.
2. You then take a step for each time step between $T$ and $1$:
3. You sample white noise $\mathbf{z}$ from the standard normal distribution. This is because we are using the [[The Reparameterization Trick]] so this is the $\epsilon$ noise for $\mathcal{N}\left(\mu, \sigma^2\right)=\mu+\sigma \cdot \epsilon$.
4. You then $x_{t-1}$ where $\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right)\right)$ is the mean of $x_{t-1}$ and $\sigma_t \mathbf{z}$ is the standard deviation multiplied by the random noise you sampled in (3).
5. You continue the process until time step $T$
6. You return $x_0$ which is the de-noised version of $x_T$.

At each timestamp the noise prediction model predicts the total noise needed to go from the image at timestamp $t$ to the image at timestamp $0$. However, we don't trust the model enough to go back to the original image. Therefore, the model still predicts the total noise but you only subtract a fraction of this (in the image below you down weight with $\epsilon$). This then takes you only a single step to $t - 1$. This allows you to go step-by-step to improve on the previous result and inject additional textual information when doing text-to-image generation (easier to inject info gradually than at once).
![[stable-diffusion-model-sampling.png]]

### Why iteratively de-noise the image? [Slack Thread](https://zoox.slack.com/archives/C0EE86ACW/p1679946673370659)
**Question**: Why does [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) directly predict the full noise added to the image via the forward diffusion process but for sampling and the reverse diffusion process you need to iteratively de-noise the image? It seems like if the noise model can directly predict the full noise added to go from `time=0` to `time=t` then you can just predict that noise and complete the reverse process in one step.

**Answer**: The initial estimate at the start of inference beginning with random noise will likely be quite bad. Empirically it seems that good diffusion processes tend to start from random noise, bumble around for a bit until a structure starts to emerge, and then refine that to produce a nice output (the first figure).

![[tutorial-on-denoising-diffusion-based-generative-modeling-20230327182329165.png]]

If you just randomly sample gaussian noise and feed it through a network to generate a sample that would fit the description of a VAE. Empirically the multiple passes of denoising and stochasticity during the inference process are important to achieving high-quality samples (typically measured by FID for image synthesis).

As another example, the right plot is from one of my training runs for a diffusion model. The x-axis is the log of how much noise is added and the y-axis is a measure of how well the single-step denoised estimate used during training matched the original sample. You can see the reconstructions are quite bad for large noise levels. The iterative diffusion inference process makes that ok (assuming the reconstruction loss eventually gets low enough for lower noise levels).

![[tutorial-on-denoising-diffusion-based-generative-modeling-20230327182343292.png|600]]

# Network Design
Most diffusion models use a U-Net architecture to take an image $x_t$ and predict the noise that was used to generate that image from $x_{t-1}$ which is $\epsilon_\theta\left(\mathbf{x}_t, t\right)$. Using a U-Net architecture is because you take an input and need to produce an output with the same spatial resolution.

![[tutorial-on-denoising-diffusion-based-generative-modeling-20230324165119039.png]]

The purple represents a residual block (ex. ResNet block) that includes a self-attention layer. The same network is used for different timesteps so you pass in the time step $t$ (represented as an embedding like sinusoidal positional embedding) through fully connected layers and then include this in the residual blocks.

You can add the time features to the residual blocks using either spatial addition or adaptive group norm layers.

**Adding text to the model**
![[add-text-concat-as-input.png|400]] ![[add-text-cross-attention.png|380]]
You can add the text embeddings to the model by:
1. Concatenating the output from a language transformer to the image input.
2. Using cross-attention in the layers of the U-Net to attend to the text tokens.

# Diffusion Parameters
### $\beta_t, \sigma^2_t$ schedule
The $\beta_t$ schedule determines how much noise is added at each timestamp and **controls the variance of the forward diffusion process**. You can often use a linear schedule where the amount of noise added grows linearly with the increased timestamp. Later papers from OpenAI changed the linear schedule to a cosine schedule. Using a cosine schedule results in the information from the noise-less image being destroyed slower over time (see picture below).
![[diffusion-linear-vs-cosine.png]]


The earlier papers set the variance, $\sigma^2_t$, equal to $\beta_t$ at each timestamp. $\sigma^2_t$ **controls the variance of the reverse denoising process**. You can also learn the variance vs. using $\sigma_t^2 = \beta_t$ which later papers from OpenAI did and saw improvements with. Note that learning these parameters will also change the loss since you now need to predict for both the mean and the variance of the distributions.



### Content-Detail Tradeoff
![[content-detail-tradeoff.png]]
For large $t$, the model is specialized for generating coarse content (ex. general shape of a cat). For small $t$, the model is specialized for generating finer details.

# Connection to VAEs
![[Autoencoders#VAE TLDR]]

Diffusion models can be thought of as a special form of VAEs. Like VAEs, they have a learned decoder that takes a distribution and then produces a distribution to sample from. Unlike VAEs, the forward encoding process is fixed and doesn't need to be learned via an encoder. [Source](https://youtu.be/fbLgFrlTnGU)
![[vae-and-diffusion-models.png|700]].

> [!QUESTION] Not really sure what this means
> Both types of models also try to maximize a lower bound on $\log(p_\theta(x))$.

![[gans-vs-vaes-vs-ddm.png]]

# Results
Diffusion Models are often evaluated using the [[FID Score]].