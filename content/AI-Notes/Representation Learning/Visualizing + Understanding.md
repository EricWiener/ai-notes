---
tags: [flashcards, eecs498-dl4cv]
source:
summary: understanding what neural networks learned.
---

[[498_FA2019_lecture14.pdf.pdf]]

We want to understand what neural networks are learning after we train it. We want to see what different features and layers look for.

1. **Saliency Maps**: Saliency maps are a quick way to tell which part of the image influenced the classification decision made by the network.
2. **Adversarial Attack**: We can perturb an input image so that it appears the same to humans, but will be misclassified by the pretrained network.
3. **Class Visualization**: We can synthesize an image to maximize the classification score of a particular class; this can give us some sense of what the network is looking for when it classifies images of that class.

# First Layer

We can visualize the first layer of the network by just visualizing the filters (weights) themselves. **We take the inner product of a filter with an image. An image that matches the filter will get a very strong response to the filter.** We can use this to visualize the filters of the first layer. 

The visualizations of the first layer are similar even for different models.

![[visualizing-first-layers.png]]

Visualizing the filters of the **first layer.**

> [!note]
> We can try to visualize filters at higher levels, but it doesn't tell us much.
> 

![[AI-Notes/Representation Learning/visualizing+understanding-srcs/Screen_Shot 1.png]]

The second layer has 20 filters. Each of the 20 filters has a depth of 16, height of 7, and width of 7. We visualize it as 20 sets of 16 7x7 greyscale images. This is very hard to interpret.

# Last Layer

We can skip what happens in the intermediate layers and can try to look at the last layer (immediately before the classifier layer that gives per-class scores). You can run the network on many images and collect the feature vectors. 

### Last Layer: Nearest Neighbors

We can look for the nearest neighbors in the feature space that is learned by the network. We can look for images that have the smallest L2 distance in the feature space. **This is looking at the activations (not the filters) now.**

![[AI-Notes/Representation Learning/visualizing+understanding-srcs/Screen_Shot_1.png]]

The neighbors found using this new feature space are much more complex similarities than just using nearest neighbors on the pixel space.

### Last Layer: Dimensionality Reduction

We can try to reduce the space of the fully connected activations from 4096 (for AlexNet) to 2 dimensions. This can be done with something like **Principal Component Analysis** (PCA) or **t-SNE** (a more complex version).

**PCA** preserves as much of the features of the high-dimensional space as possible, while also reducing it to two dimensions.

**This is looking at the activations.**

![[This is visualizing the activations for images of digits (there are 10 clusters - one for each digit).]]

This is visualizing the activations for images of digits (there are 10 clusters - one for each digit).

# Maximally Activating Patches

We can look at the filters (weights) in the middle of the network by running all the training images through the network and recording the response of the filter that was highest. We can then look at the receptive field that this corresponded to in the original image to see what patch of the original image caused the filter to activate the most.

![[visualizing-earlier-layer.png]]
Visualizing an earlier layer in the image. Each row is a different neuron.

![[visualizing-deeper-layer.png]]
Visualizing a deeper layer (larger receptive field).

# Which Pixels Matter?

We can look at which parts of the image actually matter. We can mask different parts of the image and see how much the predicted probabilities change.

![[AI-Notes/Representation Learning/visualizing+understanding-srcs/Screen_Shot 5.png]]

**Saliency:** the quality of being particularly noticeable or important

**Occlusion:** blockage or closing of something

![[AI-Notes/Representation Learning/visualizing+understanding-srcs/Screen_Shot 6.png]]

![[AI-Notes/Representation Learning/visualizing+understanding-srcs/Screen_Shot 7.png]]

You can create **saliency maps** that show how much each part of the image contribute to the prediction. If you apply the mask over a certain area, the change in probabilities is shown in the chart. It makes sense that putting the mask over the sail boat will change the probability more.

This is pretty **expensive** to do since you need to compute the forward pass for every possible position of the mask.

### Saliency via Backprop

You can also compute saliency using backprop instead of occlusion. You first compute the forward pass, and then backpropagate. You look at the gradients of the image with respect to the loss and see which pixels have the largest magnitude gradients (changing them a little will change the loss a lot).

![[AI-Notes/Representation Learning/visualizing+understanding-srcs/Screen_Shot 8.png]]

![[AI-Notes/Representation Learning/visualizing+understanding-srcs/Screen_Shot 9.png]]

This shows the gradient of the dog score with respect to the pixels.

**Most examples don't come out so nice.** The authors of the paper likely chose images that looked good.

You can even use these saliency maps to try to segment out the object without supervision. 

![[Using GrabCut on the saliency maps]]

Using GrabCut on the saliency maps

## Intermediate Features via Guided Backprop

We can apply a similar idea as before where we used backpropagation to see what pixels in the input image affected the score of the correct class the most. 

However, now we will see what input pixels affected a certain neuron the most. For instance, we will pick a single intermediate neuron, such as one value in a 128 x 13 x 13 conv5 feature map (in AlexNet). We then compute the gradient of the neuron value with respect to image pixels.

For some reason, visualizing the results works better when we use **guided backprop** instead of regular backprop. This means then when backpropping through the ReLU, we only allow positive gradients through.

![[AI-Notes/Representation Learning/visualizing+understanding-srcs/Screen_Shot_2.png]]

![[AI-Notes/Representation Learning/visualizing+understanding-srcs/Screen_Shot 11.png]]

On the left shows the maximally activating patches (each row is a different neuron). On the right, we see the maximally activating pixels for each neuron when we use guided backprop.

# Gradient Ascent

We can use **gradient ascent** to generate a synthetic image that maximizes a certain class score. 

$$
I^* = \textbf{argmax}_I f(I) + R(I)
$$

- $f(I)$ is the value of a particular class score (we want to maximize this particular score).
- $R(I)$ is a natural image regularizer (to ensure we don't just get noise).

### Algorithm

We can initialize an image to zeros or random noise.

Repeat:
- Compute the forward pass through the network with the images to compute scores (pre-softmax).
- Backprop to get the gradient of neuron values with respect to image pixels
- Make a small update to the image based on the gradient

### Natural Image Regularizer
One regularizer we can consider is using the L2 norm of the generated image. This would be: $I^* = \textbf{argmax}_I f(I) - \lambda ||I||_2^2$.

![[AI-Notes/Representation Learning/visualizing+understanding-srcs/Screen_Shot 12.png]]

You can use more complex regularizers to get even better results.

![[AI-Notes/Representation Learning/visualizing+understanding-srcs/Screen_Shot 13.png]]

> [!note]
> You can also do the same thing to maximize the firing rate of a particular neuron.
> 

### Adversarial Examples

You can use the same idea of gradient ascent to generate **adversarial images.** You can start with an image of a certain class and then use gradient ascent to modify it slightly so it becomes predicted as a different class.

![[AI-Notes/Representation Learning/visualizing+understanding-srcs/Screen_Shot 14.png]]

### Feature Inversion

You pass an image through a network and get its feature representation. Then, we try to synthesize a new image that has the same feature representation. We will also add a natural image regularizer.

This tells us how much of the original image is preserved in its feature representation.

![[reconstructing-from-different-layers-of-vgg.png]]

Reconstructing from different layers of VGG-16

The latter layers start to lose some of the local texture and color information. The more information is thrown away by the raw images in the latter layers.

### DeepDream

DeepDream was a project by Google that amplified existing features of an image.

![[AI-Notes/Representation Learning/visualizing+understanding-srcs/Screen_Shot 16.png]]

You take your original image:

- Run it through the CNN and extract features at a particular layer
- Set the gradient of the layer equal to the activation values themselves. This will cause whatever features were activates in the network to be activated even more strongly.
- Update the image

# Texture Synthesis

Given a small patch of an image, we want to generate an output image that is similar to the input patch. You can get pretty good results with just traditional approaches.

Traditional Approach:

Neural Network Approach: