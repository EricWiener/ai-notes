---
source: [[lec15-language.pdf]]
aliases: [PixelRNN, PixelCNN]
tags: [eecs442, eecs498-dl4cv, flashcards]
summary:
---

[[lec15-language.pdf]]
EECS 498 DL4CV Lecture 19 - Generative Models 

# Density Estimation
Density estimation attempts to fit a probability distribution to your input data $x$. If you were to sample points from the probability distribution function then receiving examples that you have in your training set $x$ would be a reasonable outcome.

This is unsupervised learning because you don't know the ground truth function and you don't know the probability of every point.

# Explicit Density Estimation
The goal of explicit density estimation is to find a function:
$$
p(x) = f(x, W)
$$
Where the likelihood of seeing $x$ is a function of $x$ and a learnable weight matrix $W$. 

Given a dataset $x^{(1)}, x^{(2)}, ... x^{(N)}$ we want to find the $W$ matrix that makes this data as likely as possible (which causes data not in this dataset to be less likely). We can train this model by solving to maximize the probability of the training data:
![[AI-Notes/Generative/autoregressive-models-srcs/Screen_Shot.png]]

If you assume the examples in your dataset are independent, maximizing the probability of your dataset is the same as maximizing the product of the probabilities of each individual example:
$$W^*=\arg \max _{\mathrm{W}} \prod_i p\left(x^{(i)}\right)$$
You can then use a log trick to exchange the product for a sum. Working with products of a lot of numbers is numerically unstable so you can replace the product with taking the sum of the logs of the probabilities. $\log$ is a monotonically increasing function so applying a log inside an arg-max doesn't change the result of the arg-max.
$$=\arg \max _W \sum_i \log p\left(x^{(i)}\right)$$
Given that we want to solve for $W^*$, we can make this into our loss function to find the best $W$ for our function $f$:
$$\\arg \max _W \sum_i \log f\left(x^{(i)}, W\right)$$
You can solve for $W$ via gradient descent.

### Explicit Density: Autoregressive Models
$p(x) = f(x, W)$ is the general form for explicit density estimation. For autoregressive models, we use a particular form of this function:
- We assume $x$ consists of multiple subparts $x = (x_1, x_2, ..., x_T)$. For instance, this could be an image where each subpart are the pixels
- The probability of $x$ is given by $p(x) = p(x_1, x_2, ..., x_T)$. This means $p(x$) is the probability that all the subparts of $x$ are true.
- We can break down the probability using the chain rule of conditional probability:
     $$
    p(x) = p(x_1, x_2, ..., x_T) \\ = p(x_1)p(x_2|x_1)p(x_3|x_1, x_2)... \\ = \Pi_{t=1}^Tp(x_t|x_1, ..., x_{t-1})
    $$
    
- $p(x_t|x_1, ..., x_{t-1})$ is the probability of the next subpart given all the previous subparts

> [!NOTE] We want to train a model that can predict the probability of the next subpart given all previous subparts: 

This looks a lot like RNN. When we did Language modeling with an RNN, the probability of the next token was conditioned on the probability of all the previous tokens. You don't need any training labels, you can just predict the next token conditioned on all the previous tokens.

![[AI-Notes/Generative/autoregressive-models-srcs/Screen_Shot 1.png]]

> [!NOTE] Using transformers to generate images
> All that the recurrent networks is doing is predicting the next pixel conditioned on all previous pixels. You could use masked attention with a transformer to prevent the transformer from looking ahead in the sequence. Then, you can generate one pixel at a time the same way you do with an RNN. 

# PixelRNN (generate image with RNN)
Previously we used steerable filter features to do texture synthesis (find the most similar image patch based on responses to hand-crafted filters and then choose a pixel from random from this patch when filling in another patch). However, this required hand-crafted features.

An alternative to hand-crafted features and GAN's is to use autoregressive probability models to generate textures.

PixelRNN learns a probability distribution for the colors of an image. It represents colors as discrete categories (no longer continuous values). The distribution can have multiple peaks now.

The model will predict a new color given the previous pixels.

![[AI-Notes/Generative/autoregressive-models-srcs/Screen_Shot 2.png]]

Predicting what color to use given the previous pixels. Red is the most likely color.

### Algorithm
PixelRNN will generate image pixels one at a time, starting at the upper left corner. It computes a hidden state for each pixel that depends on hidden states and RGB values from the left pixel and from above (LSTM recurrence): $h_{x, y} = f(h_{x-1, y}, h_{x, y-1}, W)$.

At each pixel, you first predict the red value, then blue, and then green out of the discrete values $[0, 1, ..., 255]$. You finally compute a softmax to choose a color.

![[AI-Notes/Generative/autoregressive-models-srcs/Screen_Shot 3.png|200]]
![[AI-Notes/Generative/autoregressive-models-srcs/Screen_Shot 4.png|200]]
![[AI-Notes/Generative/autoregressive-models-srcs/Screen_Shot 5.png|200]]
![[AI-Notes/Generative/autoregressive-models-srcs/Screen_Shot 6.png|200]]
![[AI-Notes/Generative/autoregressive-models-srcs/Screen_Shot 7.png|200]] ![[AI-Notes/Generative/autoregressive-models-srcs/Screen_Shot 8.png|200]]
Each pixel depends implicitly on all the pixels above and to the left (shown in blue). You have to provide it with the initial pixel to use. You sometimes pad the top row and left column with special pixel values.

**Training:** 
You can train the model as you would train an RNN to generate text.

### Problem
The PixelRNN has the problem that it is very slow during both training and testing. An $N \times N$ image requires $2N - 1$ sequential steps.

# PixelCNN
![[AI-Notes/Generative/autoregressive-models-srcs/Screen_Shot 9.png]]
PixelCNN is an attempt to optimize PixelRNN. You still generate image pixels from the top-left corner, but the dependency on the previous pixels is now modeled using a CNN. You convolve over the pixels to the left and above it (within the receptive field). 

Training is faster than PixelRNN. You can parallelize convolutions since context region values are known from the training images (you aren't generating these). This is different from PixelRNN where you have to train sequentially since the hidden state depends on the previous pixels (vs. with PixelCNN where it just depends on the pixels within the conv window).

However, generating is still slow during test time because you need to proceed sequentially.

### Pros and Cons
**Pros:**
- Can explicitly compute likelihood $p(x)$. You can go through all your training data and compute the likelihood of a given pixel given surrounding pixels.
- You can train the model on training data and then give it unseen, but similar images. It should assign it a high likelihood if it has learned good features.
- Can generate reasonably good samples (generate good images)

**Con:**
- Sequential generation is slow (bad at inference time).

