---
tags: [flashcards, eecs442, eecs498-dl4cv]
aliases: [GANs]
source: https://machinelearningmastery.com/what-is-cyclegan/
summary:
---

> [!NOTE] GAN Overview
> The GAN architecture is an approach to training a model for image synthesis that is comprised of two models: a generator model and a discriminator model. The generator takes a point from a latent space as input and generates new plausible images from the domain, and the discriminator takes an image as input and predicts whether it is real (from a dataset) or fake (generated). Both models are trained in a game, such that the generator is updated to better fool the discriminator and the discriminator is updated to better detect generated images.

We can do style transfer, but now we want a network that given a label, can output an image. Previously we had classifiers that took an image and gave a label, but now we do the reverse.

![[AI-Notes/Generative/generative-adversarial-networks-srcs/Screen_Shot.png]]

We can think of the neural networks as **distribution transformers**. We start with an initial multi-variable gaussian distribution (noise). Then, we train the network to give us a synthesized image. 

![[AI-Notes/Generative/generative-adversarial-networks-srcs/Screen_Shot 1.png]]

![[AI-Notes/Generative/generative-adversarial-networks-srcs/Screen_Shot 2.png]]

We want different points in our input space to map to different images. For instance, if you shifted the input point, you would generate a duck instead of a fish.

# Generate Adversarial Networks (GANs)
We can use GANs to accomplish this goal of having different points in our input map to different images. With a normal GAN, you input a noisy image and you get a generated picture, but don't really have a say over what the result looks like. With a conditional GAN, you can input both a noisy image and what you want the output to look like, so you can control the result.

![[AI-Notes/Generative/generative-adversarial-networks-srcs/Screen_Shot 3.png]]

We will now have two networks:
- **Generator (G):** will generate synthesized images
- **Discriminator (D)**: will predict whether an image is real or fake
- The two networks will compete with each other. G tries to fool D. D tries to identify the fakes.

While **autoregressive models** and **variational autoencoders** try to model the training data $p(x)$, GANs don't try to do this. We can no longer find out how likely a specific image is. However, we will still be able to create samples. 

### Discriminator
The discriminator will be trained on both real and fake images (typically half real and half fake). It is trying to predict whether an image is fake or real by maximizing:

$$
\text{argmax}_D \mathbb{E}_{z,x}[\log D(G(z)) + \log(1-D(x))]
$$

- Here we are saying that something being a fake is 1 and being real is 0.
- $\log D(G(z))$ takes the fake image $G(z)$ and it wants to output a larger value (close to 1.0) to maximize this.
- $\log(1-D(x))$ takes a real image $x$ and it wants to output a small value (close to 0.0) to maximize this.
- $\mathbb{E}$ is the expected value (we are considering performance on average over all the values in our dataset).
- We choose the discriminator model $D$ that maximizes this expression

### Generator
The generator is trying to fool **D.**

$$
\text{argmin}_G \mathbb{E}_{z,x}[\log D(G(z)) + \log(1-D(x))]
$$

This is the same equation the discriminator was trying to maximize, but now we are trying to minimize it with respect to $G$. This means we want the discriminator to do as poorly as possible.

### Generator as Mini-Max Problem
We can combine the two expressions into a **mini-max** equation (has a min and a max). We are trying to find the $G$ that minimizes the best $D$.

$$
\arg \min_G\max_D \mathbb{E}_{z,x}[\log D(G(z)) + \log(1-D(x))]
$$

![[AI-Notes/Generative/generative-adversarial-networks-srcs/Screen_Shot 4.png]]

We will alternate between training the generator and discriminator (alternating gradient descent). The global optimum is when G reproduces the data distribution (in practice usually the discriminator wins).

Very best $G$ performance you can get is 50% because the discriminator will only be able to take a chance guess at what is real or fake.

### Optimality
The global minimum of the minimax game happens when

1. $D^*_G(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_G(x)}$ (Optimal discriminator for any G)
2. $p_G(x) = p_{\text{data}}(x)$ (Optimal generator for optimal D)

**Caveats:**
- The generator and discriminators are neural networks with fixed architectures. We don't know whether they can really **represent** the optimal $G$ and $D$, which can be any function.
- This tells us nothing about **convergence** to the optimal solution.

### Training
We alternate between taking gradient descent steps on the discriminator and the generator. Each has their own loss function they are trying to minimize. However, these loss functions are also intertwined. It doesn't matter if the discriminator has a very low loss, if the generator is terrible and has a high loss (it's not very difficult to detect very badly faked images). Therefore, the loss function will often bounce around a lot during training.

**Problem: vanishing gradients for G**

Additionally, when you first start training, the generator is very bad and the discriminator can easily tell apart real/fake, so $D(G(z))$ will be close to 0. This can lead to vanishing gradients for $G$, since all its outputs have the same value of 0. At the very beginning of training, the **generator** won't get updated a lot.

Currently, G is trained to minimize $\log(1 - D(G(z))$. Instead, we will train it to maximize $-\log(D(G(z))$. This has the same affect of trying to make the generator make realistic images, but it will also get strong gradients at the start of training..

![[AI-Notes/Generative/generative-adversarial-networks-srcs/Screen_Shot 5.png]]
The gradient at the start of training will be larger for  $\log(D(G(z))$

### Mode collapse
Mode collapse happens when the generator can only produce a single type of output or a small set of outputs. This may happen due to problems in training, such as the generator finds a type of data that is easily able to ==fool the discriminator== and thus keeps generating that one type.
<!--SR:!2024-12-20,419,310-->

A GAN is successfully trained when both of these goals are achieved
1. The generator can reliably generate data that fools the discriminator. 
2. The generator generates data samples that are as diverse as the distribution of the real-world data.

Mode collapse happens when the generator fails to achieve Goal #2–and all of the generated samples are very similar or even identical.

The generator may “win” by creating one realistic data sample that always fools the discriminator–achieving Goal #1 by sacrificing Goal #2.

### Trivia
You can even find a transition between two images by looking for the images generated between two points in your **latent space** (input) and seeing where they end up in your **data space** (generated images).

![[AI-Notes/Generative/generative-adversarial-networks-srcs/Screen_Shot 6.png]]

You can also do vector math with the sampled images from the GAN.  You draw a lot of samples from the GAN and you assign labels to them (smiling woman, neutral woman, etc.). Then, you take the $z$ vectors (encoded vectors) for these images, average them, and then you can do arithmetic.

**Note:** the $z$ vector here are the points you randomly sample from in the Latent space (Gaussian noise). They aren't the embedded features like they were for variational autoencoders.

![[AI-Notes/Generative/generative-adversarial-networks-srcs/Screen_Shot 7.png]]

# Image Translation (Pix2Pix)

![[AI-Notes/Generative/generative-adversarial-networks-srcs/Screen_Shot 8.png]]

Image translation is when you have an input image and you want to manipulate it (rather than just generating a new similar image).

Our training data would be a picture of a street map and its corresponding satellite photo.

![[AI-Notes/Generative/generative-adversarial-networks-srcs/Screen_Shot 9.png]]

An initial attempt to solve this problem might just be to predict an image that minimizes the L1 loss between the predicted image and the actual image $||G(x) - y||_1$. However, we will end up with a blurry result.

This is because it's hard to know exactly what the green represents (it could be grass, trees, a dead lawn, etc.). Therefore, the model can minimize the loss by just making it blurry.

![[AI-Notes/Generative/generative-adversarial-networks-srcs/Screen_Shot 10.png]]

We want to have a single sample of an image. Not just the average of the most likely image. We can use a generator and discriminator (GANs) to try to get better result images. However, we don't just want the generator to give a realistic image, **we want it to be a realistic image that is based on the input Google map.**

Therefore, we can concatenate the original input and the generated image as the input to the discriminator (3 + 3 = 6 channels). We now condition the GANs loss using the input image as one of the parameters (shown in purple below).

![[AI-Notes/Generative/generative-adversarial-networks-srcs/Screen_Shot 11.png]]
Conditional GANs model

We will also add an additional L1 loss between the generated image and the original image. This is because the GANs model can be a bit unstable and is harder to train. Adding this term will help stabilize training and lead to faster convergence. L2 will penalize the errors too strongly.
![[AI-Notes/Generative/generative-adversarial-networks-srcs/Screen_Shot 12.png]]

# Conditional GANs: Spectral Normalization

![[AI-Notes/Generative/generative-adversarial-networks-srcs/Screen_Shot 18.png]]

You can get a generated image for a specific label if you use a conditional GAN. This implementation used different learnable shift and scale parameters for batch normalization. This technique is called **conditional batch normalization.**

![[AI-Notes/Generative/generative-adversarial-networks-srcs/Screen_Shot 19.png]]

Now, you can say you want an image of a certain type. A normal GAN would just give you a result without you being able to say what type you wanted. Pix2Pix required a similar image to be inputted. This just requires a label.

# Trajectory Prediction
GANs can be used to predict any type of information. This paper by Professor Johnson inputted the recent trajectory of people and it would then predict where they would go.

![[AI-Notes/Generative/generative-adversarial-networks-srcs/Screen_Shot 20.png]]