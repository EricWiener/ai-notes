---
tags: [flashcards, eecs442, eecs498-dl4cv]
source:
summary:
---
[[lec12-imagesynth_(4).pdf.pdf]]

# Capturing Feature Correlations

Instead of using hand-crafted features, we can use the image features extracted by a CNN. This is a mode advanced approach than autoregressive probability models.

![[visualizing-cnn-weights.png]]
Weights for CNN look similar to hand-crafted feature detectors

We can do this by just removing the fully connected layer at the end of a CNN.

We want to know how correlated the different features (different filters) are with each other. This is a way of capturing texture. For instance, we can look at how often a horizontal filter and a vertical filter are firing together.

![[gram-matrix.png]]
Gram matrix

We can use the **gram matrix** to attempt to do this (similar to the covariance matrix). For every pair of filters, we look over all the corresponding $(x, y)$ entries in the filters and multiply them together. We add up all these products and it gives us the value for a specific filter pair.

This is a way of capturing how often filters fire together, but it also ends up discarding spatial position, since it just adds up everything.

If one filter fires high (0.9) and the other filter fires high (0.9), you will get a large value (0.9 * 0.9). However, if one filter fires high (0.9) and the other fires low (0.1), you will get a small value. You will get an even smaller value if both fire low (0.1 * 0.1).

$G_{ij}$ tells us the correlation between filter $i$ and filter $j$.

# Generating a new image

We can now use the gram matrix to try to generate a similar image to another.

![[AI-Notes/Concepts/style-transfer-image-generation-gram-matrices-srcs/Screen_Shot 2.png]]

![[AI-Notes/Concepts/style-transfer-image-generation-gram-matrices-srcs/Screen_Shot 3.png]]

For instance, given the sample image to the left, we want to emulate the image on the right. We can do this by trying to make the gram matrices for both images as similar as possible.

$$
\hat{I} = \sum_{i=1}^{128}\sum_{j=1}^{128} (G_{ij}(I) - G_{ij}(\hat{I}))^2
$$

Where $\hat{I}$ is the new image and we sum over the $x$ and $y$ positions of both images. We can now try to minimize this equation to find the optimal $\hat{I}$ using gradient descent to modify the value of $\hat{I}$.

We keep the weights of the network constant, and now backprop to the input.

> [!note]
> You can use the gram matrices from multiple feature layers of the CNN to get better results.
> 

![[AI-Notes/Concepts/style-transfer-image-generation-gram-matrices-srcs/Screen_Shot 4.png]]

![[AI-Notes/Concepts/style-transfer-image-generation-gram-matrices-srcs/Untitled.png]]

As we add on layers latter on in the network (pool4 is later than conv1_1), we get better results. This is because the later layers have larger **receptive fields** (number of pixels in the original image each convolution of a filter sees), so they get a better sense of the image.

Also note as we move from right to left, we are adding on additional layers. Therefore, the result of pool2 actually is the combined result of conv1_1, pool1, and pool2.

# Style Transfer

If we wanted to match the style of a painting, but have the result look similar to another image, we can do this.

![[AI-Notes/Concepts/style-transfer-image-generation-gram-matrices-srcs/Screen_Shot 5.png]]

We start with a synthesized random image and then need to first match the style of the painting as we did before. This is still done by minimizing the distances between the gram matrices.

However, if we just did this, we would get a random looking version of the original painting. Instead, we need to add on an additional constraint that the **content** of the synthesized and original photo are similar. This is done by using a distance calculation in the **feature space**.

$$
\sum_i\sum_{x, y}(c_i(x, y) - \hat{c}_i(x, y))^2
$$

Where $c_i(x, y)$ is a particular feature activation at a $x, y$ coordinate of the original image and $\hat{c}_i(x, y)$ is a particular feature activation at a $x, y$ coordinate of the synthesized image. This is called the **content loss.** 

**We sum the two losses together to get the final loss we minimize.**

Because we are using **gradient ascent** and need to keep doing forward/backward passes to update the original image. The process is actually very slow.

### Fast Style Transfer

You can get faster style transfer by training a neural network to output stylized images. The input to the network is the original image and the corresponding target for it is the image stylized with the slower gradient ascent based approach. Training will take a while, but you will end up with a network that can generate images very fast. Professor Johnson did this and it got used by Snapchat, Google, Facebook, etc.

The model created by Professor Johnson could only apply a single style to an image. If you wanted to be able to apply multiple styles, you would need a whole new model.

> [!note]
> Instance normalization was actually developed as a way to get fast results while doing style transfer.
> 

Another paper came out with an approach that used **conditional instance normalization**. You would learn a separate scale and shift parameter for each style for the instance normalization. This was enough to be able to apply different styles and blend the styles together.