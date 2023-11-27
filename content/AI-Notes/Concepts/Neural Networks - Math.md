---
tags: [flashcards, eecs445, clean-up]
source:
summary: The math behind training a neural network.
---

## Slides
[[eecs445-L14-training-neural-networks.pdf]]

## Overview
Goal: automatically find useful features for our learning algorithms

The main difference between machine learning and deep learning is in machine learning, you need to perform feature extraction yourself.

However, it is sometimes useful to separate features that are useful in pre-processing. This is also useful because the features you choose are interpretable.

Neural networks and deep learning are the same thing. Deep learning is just for neural networks with more than one hidden layer.

# Terminology
![[neural-network-terminology.png]]


- The hidden layers are called the architecture. Hidden layers are any layers that aren’t the input or output.
- The output function is denoted $h(\bar{x}, W)$. This is in terms of the input $\bar{x}$ and all the weights $W$.
- Weights aren’t necessarily normalized. They don’t have to add up to 1.
- Dense Neural Networks
    - Connections in one direction
    - Directed acyclic graph
    - Function of current input
    - Has no internal state other than weights
- Recurrent Network
    - Feeds output back into own input
    - Activation forms dynamical system
    - Response to input depends on current state and inputs

- **Perceptron**
![[neural-networks---math-20220319183346765.png|300]]
- This is the simplest possible NN where you have the inputs directly connected to the outputs.
- Each output has its own network, so you could learn the weights for the inputs to each output in parallel.
- Perceptron can’t learn XOR - linear functions aren’t enough to cut it
- Perceptron Algorithm won’t converge if the data isn’t linearly separable

# Single Layer NN
![[neural-networks---math-20220319183607687.png|400]] ![[linear-layer-diagram.png|400]]


You have your inputs and they each map to a node in the next layer. Each connection has a weight corresponding to it. At each node:
- First you take a weighted sum of the inputs (a linear function of the inputs). $\sum_{j=1}^d w_jx_j + w_0=z$.
- Then apply a non-linear activation function $g$ or a non-linear transformation. $h = g(z)$.

## Dimensionality

- The number of neurons in the input layer is determined by the input.
- The number of neurons in the output layer is determined by your labels.
- The number of neurons in the hidden layers are up to you

**Fully connected:** each node in this layer is connected to all nodes from the previous layer. Fully connected can refer to either an entire network or a specific layer.

**Hidden layers** allow us to learn a linear classifier in a different feature space.

**Layer number** both the nodes and the weights will have a number corresponding to their layer number.

## Fully Connected
![[fully-connected-neural-net.png]]
You can use multiple layers of neurons to create a fully connected neural network. The non-linear activation functions keeps everything from collapsing to a single linear operation.

**Note that the circles are the inputs and outputs. The crossed lines are the weights.** The weights are what the model learns.
![[one-layer-nn-learned-seperation.png|300]] ![[two-layer-nn-learned-seperation.png|300]]


A two layer network (right) can separate data that was inseparable with a single layer network (left).

## Activation Functions

Want them to be differentiable. The ReLu works very well (rectified linear unit). $f(z) = \max(0,z)$. You use the same activation function across all the neurons in a specific layer.

Hard threshold and sigmoid threshold
![[hard-threshold-and-sigmoid-threshold.png]]


Hyperbolic tangent and rectified linear unit (ReLU)
![[hyperbolic-tangent-and-relu.png]]



## Naming Conventions
- The first hidden layer is typically called the second hidden layer of the network.
- A weight is denoted $w^{(a)}_{bc}$
    - $a$ is the layer the weight belongs to (same # as the layer of nodes it comes from)
    - $b$ is the node # it connects to.
    - $c$ is the node # it comes from.

# Computing Gradient in SGD
## Algo
- Initialize $\bar{\theta}$ to small random values.
    - If you use the same constant value, your updates won’t change anythings.
- Randomly select a training instance $(\bar{x}^n, y^{(n)})$
    - Make a prediction $h(\bar{x}^{(n)}, \bar{\theta})$
    - Called forward propagation
    - Measure the loss of the prediction $h(\bar{x}^{(n)}, \bar{\theta})$ with respect to the true label $y^{(n)}$
        - Calculate Loss $(y^{(n)}h(\bar{x}^{(n)}, \bar{\theta}))$
        - This is supervised learning, so we have the correct label
    - Go through each layer in reverse to measure the contribution of each connection. **Backward propagation.**
    - Tweak weights to reduce error

## Backward Propagation
Don’t want to keep having to retake the gradient from scratch for every weight. In the past, this is what made neural networks so difficult to implement. Backward propagation is a way to organize the partials.

![[forward-and-backward-pass-math.png]]

![[forward-backward-pass-math-2.jpg]]

## Softmax and Cross Entropy Loss
Use the **softmax** activation function to output a probability distribution after the final layer. This is calculated with:

$\hat{y}_c = \frac{e^{z_c}}{\sum_j e^{z_j}}$

It has the nice property that it can take in negative values and ensure the output is positive and that all the outputs add up to 1 (converts everything to probabilities).

**Cross Entropy Loss** measures the difference between the true probability distribution and the calculated one. This is needed when we have multiple labels we are predicting (no longer binary classification).

Compare $\hat{y}_c$ with the ground truth vector $\bar{y}$ using cross entropy loss. $\mathcal{L}(\bar{y}, \hat{y}_c) = -\sum_c y_c \log \hat{y}_c$

![[example-cross-entropy-loss-calculation.jpg]]

In the above example, we have three different values for labels (1, 2, 3). Therefore, our neural network will need to have three different output nodes that each give a probability that the datapoint should be labeled 1, 2, or 3. These outputs are given as logits and in our particular example, the logits are [2, 4, 0.5]. We need to use the soft-max on these logits to convert them to probabilities. Then we can calculate the cross-entropy loss.

## Binary Cross Entropy

You use this when you are predicting a binary label. It tries to tell you how much information you need to get from your predicted values to the actual labels.

$h_w(X)$: [1, 0.3, 0.8] → can take a range of values

$y$: [1, 1, 0] → either 1 or 1

$Loss(W) = \frac{1}{n} \sum_i y^{(i)}\log(h_W(x^{(i)})) + (1 - y^{(i)})\log(1 - h_W(x^{(i)}))$ Equation explained:

- $\frac{1}{n}$: need to normalize by number of data points to make sure that you don’t make the loss higher for models with more training points.
- $y^{(i)}\log(h_W(x^{(i)}))$: if y is 1, the first term will be non-zero. $$\log(1)$$ is 0, so if you correctly predict $\hat{y}^{(i)}$, your error will be 0
- $(1 - y^{(i)})\log(1 - h_W(x^{(i)}))$: if y is 0, the first term will be non-zero. If you are correct, $\log(1 - 0) = \log(1) = 0$, so you will have no error.

# Choice of Architecture
- Number of hidden layers
    - A neural network with a single hidden layer can approximate any continuous function (within an $\epsilon$ value) given it has enough neurons.
        - 2 layers NNs can approximate any function (the layers would need to be infinitely wide)
        - You need $n^2$ connections for two fully connected layers (both of size $n$), so the wider the layer, the more expensive.
    - Deeper nets can converge faster to a good solution
    - Start with 1-2 hidden layers and ramp up until you start to overfit
- Number of neurons per hidden layer
    - # nodes in input and output are determined by the task at hand
        - d + 1 in input (because of bias)
    - Common practice: same size for all hidden layers (one tuning parameter)
    - Increase number of neurons until you start to overfit
- How to set learning rate
    - Simplest way: keep it fixed and use the same for all parameters
        - Better results can generally be obtained by allowing learning rates to decrease (learning schedule)
- Reducing Overfitting: Neural networks are susceptible to overfitting
    1. Early stopping: interrupt training when performance on the validation set starts to decrease
    2. L1 & L2 regularization: applied as before to all weights
    3. Dropout

## Dropout
![[nn-after-dropout.png]]
- Can reduce overfitting with dropout
    - At each training iteration, drop some nodes from graph
    - Prevents relying completely on one feature
    - At test time, you keep all nodes and use a weighted sum of the nodes
    - Conceptually similar to ensemble approach → each training iteration configuration is its own classifier
- Make sure to disable dropout when doing final classification

See this page for more details:

## Early Stopping
Stop when the training error and testing error start moving in opposite directions (training error decreases and testing error increases)

![[early-stopping.png]]

## Vanishing/Exploding Gradients
- The gradient in deep neural networks is unstable, tending to either explore or vanish in earlier layers
    - Sometimes gradients get smaller and smaller as the algorithm moves towards the lower layers. This can leave the lower layers virtually unchanged and training never converges to a good solution
- Less of a problem when we use ReLU
- Some causes of vanishing gradients:
    - Choice of activation function
    - Multiplying by many small numbers (weights initialized poorly)

## Adversarial Machine Learning: Fast Gradient Sign Attack
Adversarial machine learning attempts to try to trick models that have already been learnt. One approach is the fast gradient sign attack.

Rather than working to minimize the loss by adjusting weights based on the back propagated gradients, the attack adjusts the input data to maximize the loss based on the same back propagated gradients.

$\text{adv-x} = x + \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))$
-   adv-x: the new adversarial image
-   x: the original image
-   y: original image label
-   $\epsilon$: multiplier to ensure the perturbations are small
-   $J$: loss
-   $\theta$: model parameters
-   $\epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))$: noise to be added to the original image

Example: adding noise to an image of a panda to get a classification as a gibbon.