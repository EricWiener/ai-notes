Tags: EECS 498 F20

[[598_FA2020_lecture05.pdf]]

Linear classifiers aren't that powerful. Not all relationships have a linear separation (most don't). They only have one template per class.

## Solution #1: Feature transform

![[AI-Notes/Concepts/Neural Net/Screen_Shot.png]]

Transform the data so it is linearly separable. This was done more often before neural networks became more common.

> [!note]
> An SVM with a non-linear kernel is basically a linear classifier, but it just transforms the data. Typically the kernel is not learned in response to training.
> 

## Solution #2: Use features instead of raw image

### Image Features: Color Histogram

Just use the number of each color (for example)

![[AI-Notes/Concepts/Neural Net/Screen_Shot 1.png]]

### Image Features: Histogram of Oriented Gradients (HoG)

1. Compute edge direction / strength at each pixel. 
2. Divide image into 8x8 regions. 
3. Within each region compute a histogram of edge directions weighted by edge strength.

![[AI-Notes/Concepts/Neural Net/Screen_Shot 2.png]]

Example: 320x240 image gets divided into 40x30 bins; 8 directions per bin; feature vector has 30*40*9 = 10,800 numbers.

This is useful because it can handle changes in color or shade. Now we just keep the spatial information (while color histogram threw away spatial information and just kept color information).

### Image Features: Visual Bag of Words (Data-Driven!)

![[AI-Notes/Concepts/Neural Net/Screen_Shot 3.png]]

Extract random patches from the images. Then encode the images with how many of each type of patch they have. Then we can predict by looking at which patches an image has and finding the nearest class.

### Image Features: Overview

You can concatenate multiple image features. Don't just have to choose one. For example, you could combine a histogram of oriented gradients, a color histogram, and a histogram of patch encodings.

However, with this approach of first extracting features and then using a linear classifier, you can only change the weights of the classifier. You can't change how you extract features.

# Neural Network

A neural network basically does feature transformations up until the last layer, where a linear classifier is used to give predictions. 

> [!note]
> Neural Networks allows us to modify the feature extraction and weights of the classifier to lower the loss, while with previous methods, we could only modify the weights of the model.
> 

### Math

![[AI-Notes/Concepts/Neural Net/Screen_Shot 4.png]]

Now we have more weights and biases. The two-layer neural network first performs a feature transform and then it is just acting like a linear classifier.

This approach generalizes to arbitrary layers of complexity.

![[AI-Notes/Concepts/Neural Net/Screen_Shot 5.png]]

The hidden layer is the **transformed features** (note it refers to the features not the weights) after the first weights are applied. 

A fully-connected neural network (aka "Multi-Layer Perceptron") is called this because weights in one layer will affect future layers because they transform the features.

A two-layer neural network can allow us to have different templates for a single class (can recognize a horse facing both directions).

A neural network can be trivially extended to be any depth. Easy to create **deep neural networks.** The **width** of a network is the length of a particular layer. The **depth** is the number of layers (usually we count the number of weight matrices - ex. this is 6 deep).

![[Here the gray columns show values (input, intermediate outputs, and the final scores) and the white columns show the weight matrices of the layers.]]

Here the gray columns show values (input, intermediate outputs, and the final scores) and the white columns show the weight matrices of the layers.

Multiple layers in a deep neural network help with classifications because you can compute features, recombine features, and keep recombining features to do more computation. This allows you to learn more non-linear decision boundaries. The number of layers can perform more non-trivial computation.

A single layer network can represent any function, but its width would need to be almost infinitely wide. A deeper, but shallower network can do a better job with fewer weights needed.

## Universal Approximation

A neural network with one hidden layer can approximate any function $f: R^n \rightarrow R^m$ with arbitrary precision*. A neural network with one hidden layer with infinite width can approximate any function.

**Be aware:**

- Neural networks can represent any function
- Don't know whether we can actually learn any function with SGD
- Don't know how much data we need to learn a function
- Remember: kNN is also a universal approximator (can approximate any function with infinite data).

## Convex Functions

A function is convex if it is shaped like a high dimensional bowl. Convex functions are **easy to optimize**: can derive theoretical guarantees about **converging to global minimum**.

Linear classifiers optimize a **convex function** (SVM and Softmax). This means linear classifiers (under some assumptions) can converge to a global minimum.

However, Neural Networks don't have the same nice theoretical guarantees because they aren't guaranteed to be convex. They need **nonconvex optimization.** In practice, they seem to converge nicely anyways.