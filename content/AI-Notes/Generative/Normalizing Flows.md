---
tags:
  - flashcards
source: https://ar5iv.labs.arxiv.org/html/1807.03039
summary:
---
[Medium Article](https://towardsdatascience.com/introduction-to-normalizing-flows-d002af262a4b)
[1 hour long youtube video](https://youtu.be/u3vVyFVU_lI)
[15 minute Ari Seff video](https://youtu.be/i7LjDvsLWCg)

What's the difference between [[Generative Adversarial Networks|GANs]] and normalizing flows?
??
GANs are trained to take a random vector (ex. sampled from Gaussian noise) and produce a data point (ex. an image) via a generator that produces images and a discriminator that judges the quality of these. Normalizing flow models are trained to take a data point (ex. an image) and produce a simple distribution (ex. Gaussian) that minimizes the log-likelihood of the probability of the transformed samples. At inference time for GANs you use the same procedure as for training and sample noise and then pass it to a generator to produce an image. For normalizing flow models, the entire model is invertible so **you use a reverse process to what you used during training** and sample from noise to produce an image.
![[normalizing-flows-20231022180012008.png|650]]
<!--SR:!2023-11-01,4,270-->

Different from autoregressive model and variational autoencoders, deep normalizing flow models require specific architectural structures.
1. The input and output dimensions must be the same.
2. The transformation must be invertible.
3. Computing the determinant of the Jacobian needs to be efficient (and differentiable). This is due to the change of basis formula. See [youtube](https://youtu.be/i7LjDvsLWCg?t=93) for more details.
### Math
In simple words, normalizing flows is a series of simple functions which are invertible, or the analytical inverse of the function can be calculated.

> [!NOTE] What is an invertible function
> For example, $f(x) = x + 2$ is a reversible function because for each input, a unique output exists and vice-versa whereas $f(x) = x^2$ is not a reversible function because both $9$ can correspond to either 3 or -3. The inverse of $f$ exists if and only if $f$ is a [[Bijective Function]] (maps each input to exactly one output and vice-versa).

**Let $\mathbf{x}$ be a high-dimensional random vector with an unknown true distribution $\mathbf{x}\sim p^{*}(\mathbf{x})$. We collect an i.i.d. dataset $\mathcal{D}$, and choose a model $p_{\theta}(x)$ with parameters $\theta$.**
In the context of a dataset of images, $\mathbf{x}$ would represent a high-dimensional vector that encodes the pixel values of an image. Each element of the vector corresponds to a pixel in the image, and the dimensionality of the vector is equal to the total number of pixels in the image. The true distribution $p^*(\mathbf{x})$ would represent the distribution of all possible images that could be generated from the dataset (this is unknown), and the goal of a flow-based generative model would be to learn a model $p_{\theta}(x)$ that can generate new images that are similar to the images in the dataset.

The dataset $D$ being i.i.d. means it is "independent and identically distributed." In the context of an image dataset, it means that the presence of one image doesn't affect the probability of the next image and each image has the same likelihood. [Stats Stack Exchange Post](https://stats.stackexchange.com/questions/488041/independent-and-identically-distributed-data-images#:~:text=If%20your%20population%20is%20all,it%20has%20very%20many%20animal).

**In case of discrete data $\mathbf{x}$, the log-likelihood objective is then equivalent to minimizing:**
$$\mathcal{L(D)}=\frac{1}{N}\sum_{i=1}^{N}-\log p_{\theta}(\mathbf{x}^{(i)})$$
This is taking the average (summing over all examples and then dividing by the number of examples $N$) of the negative log of the probability that your learned model produces training example $\mathbf{x}^{(i)}$.

Optimization is done through stochastic gradient descent using minibatches of data.

**In most flow-based generative models the generative process is defined as:**
$$\begin{array}{r}{\mathbf{z}\sim p_{\theta}(\mathbf{z})}\\ {\mathbf{x}=\mathbf{g}_{\theta}(\mathbf{z})}\end{array}
$$
where:
- $\mathbf{z}$ is the latent variable
- $p_{\theta}(\mathbf{z})$ has a simple density (ex. a 0-1 normal distribution).
- $\mathbf{g}_{\theta}(\ldots)$ is an invertible (aka bijective) function that to produce a latent variable $\mathbf{z}$, you compute ${\bf z}={\bf f}_{\theta}({\bf x})={\bf g}_{\theta}^{-1}({\bf x})$. This means $\mathbf{f}_\theta$ is the inverse of $\mathbf{g}_\theta$.

> [!QUESTION] What is a latent variable?
> A latent variable, in the context of statistics and data analysis, is a variable that is not directly observed but is inferred or estimated from other observed variables.

During the training process of the model, you take an image $\mathbf{x}$ and then pass it to the inverse of $\mathbf{g}_\theta$ which is $\mathbf{f}_\theta$ (you are learning the model $\bf f_\theta$ during training). ${\bf f}_{\theta}({\bf x})$ then produces $\mathbf{z}$ which is a vector that belongs to the simple density function $p_{\theta}(\mathbf{z})$ you are trying to learn. For instance, if your density function has dimension = 256, then $\bf f_\theta$ will produce a vector of dimension = 256 that corresponds to the images representation in the density function space.

**We focus on functions where $\bf f$ (and, likewise, $\bf g$) is composed of a sequence of transformations $\mathbf{f}=\mathbf{f}_{1}\circ\mathbf{f}_{2}\circ\cdot\cdot\cdot\circ\mathbf{f}_{K}$ such that the relationship between $\bf x$ and $\bf z$ can be written as:**
$$\mathbf{x} \stackrel{\mathbf{f}_1}{\longleftrightarrow} \mathbf{h}_1 \stackrel{\mathbf{f}_2}{\longleftrightarrow} \mathbf{h}_2 \cdots \stackrel{\mathbf{f}_K}{\longleftrightarrow} \mathbf{z}$$
This just means that you compose a series of invertible functions in a sequence such that you can take $\bf x$ to $\bf z$ in a way that you can then perform the inverse computation.

> [!NOTE] Such a sequence of invertible transformations is also called a (normalizing) flow

**Under the change of variables formula, the probability density function (pdf) of the model given a datapoint can be written as:**

$$\begin{aligned} \log p_{\boldsymbol{\theta}}(\mathbf{x}) & =\log p_{\boldsymbol{\theta}}(\mathbf{z})+\log |\operatorname{det}(d \mathbf{z} / d \mathbf{x})| \\ & =\log p_{\boldsymbol{\theta}}(\mathbf{z})+\sum_{i=1}^K \log \left|\operatorname{det}\left(d \mathbf{h}_i / d \mathbf{h}_{i-1}\right)\right|\end{aligned}$$

which means that the log probability of a datapoint $p_{\theta}(x)$ is equivalent to something you can represent in terms of the log probability of the simple distribution of your latent variable $p_{\theta}(\mathbf{z})$.

**You can therefore represent the log-likelihood objective you are minimizing as:**
$$\mathcal{L(D)}=\frac{1}{N}\sum_{i=1}^{N}-\log p_{\theta}(\mathbf{x}^{(i)})$$
is equivalent to:
$$\mathcal{L(D)}=\frac{1}{N}\sum_{i=1}^{N}-=\log p_{\boldsymbol{\theta}}(\mathbf{z})+\sum_{i=1}^K \log \left|\operatorname{det}\left(d \mathbf{h}_i / d \mathbf{h}_{i-1}\right)\right|
$$
which is useful since you can find out the probability of a point from a normal distribution while you couldn't find out the probability of an image ($p_{\theta}(\mathbf{x}^{(i)})$) so you can now actually calculate your objective.