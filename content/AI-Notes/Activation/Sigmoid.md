---
summary: Flat end kills the gradients, outputs aren’t zero-centered, and expensive to compute
---
![[sigmoid-graph.png.png]]

Sigmoid is one of the most common activation functions that has been used for a long time. It squashes numbers into the range [0, 1].

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

![[sigmoid-math.png.png]]

It is historically popular since they have a nice interpretation as a saturating "firing rate" of a neuron. It also can be interpreted as a probability since it limits outputs to the range [0, 1].

> [!note]
> Sigmoid has multiple problems and should never be used.
> 

## **Problems:**

**Problem 1: Flat ends kill the gradient (most problematic)**

The flat ends at either end of the sigmoid **kill the gradients** and make it hard to train networks. At the flat parts, the gradient is close to zero. Having a local gradient close to zero will drastically shrink the upstream gradient during back-prop (when you multiply local gradient by upstream gradient to get downstream gradient). This is especially an issue for deep networks because it means many layers won't get a gradient propagated back to them and will stop learning.

**Problem 2: Outputs aren't zero-centered**

Sigmoid outputs are not zero-centered (the outputs are all positive). If you have a deep neural network with sigmoid activations throughout, that means every layer is receiving only positive values. The output of one layer is given by $f(\sum_i w_i x_i + b)$. When computing the gradient with respect to the weights, the gradient will be $x$, which is always going to be positive.

The upstream gradient could be positive or negative (the final loss is given by something like `softmax` which could be positive or negative). If the upstream gradient is positive, all the gradients throughout the network will be positive. If the upstream gradient is negative, all the gradients will be negative. This means all gradients will have the same sign. 

This can slow down finding optimal weights a lot. If you look at the diagram, if the optimal path to the best weights is along the positive x-axis and down the y-axis, a network with sigmoid will only be able to make movements in either (+, +) or (-, -), but won't be able to make the ideal (+, -) movement.

This gets much worse if you have weights with thousands or millions of dimensions (vs. just two as shown in the diagram) because you will always be limited to just a single one of the quadrants for a particular step.

**Note that this only applies to a single training example. Minibatches can result in having mixed gradients. Additionally, if you have a momentum term when calculating your update step, this doesn’t necessarily apply.**

![[sigmoid0-allowed-grad-update-graph.png.png]]

**Problem 3: `exp()` is computationally expensive**

It is a complicated function to compute and can take multiple clock cycles. This is more of a problem for mobile and CPU vs. GPU. On a GPU most of the time is spent moving data between memory.

It is more expensive than something like an addition or a max. However, the computational cost of an activation layer is relatively small compared to the conv layer, so not a huge issue.