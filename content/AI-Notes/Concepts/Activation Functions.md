---
tags: [flashcards]
source:
summary: Why we need activation functions + comparison plots of common ones.
---

![[relu.png]]

We need an activation function for a neural network. If we don't include it, we will just get a linear classifier (just a bunch of matrices multiplied together) no matter how many layers we have.

For instance, if we build a network with the architecture $s=W_{2} W_{1} x$, you can re-express this with:

- $W_{3}=W_{2} W_{1} \in \mathbb{R}^{C \times H}$
- $s=W_{3} x$

And your two weight matrices end up simplifying into just a linear classifier. The presence of an activation function is what makes the neural network work at all. You have an infinite number of solutions if you remove the ReLU. This isn't good.

Example:
- If your original function (with ReLU) is $f(x)=W_{2} \max \left(0, W_{1} x+b_{1}\right)+b_{2}$
- If you remove the ReLU, you end up with $f(x)=W_{2}\left(W_{1} x+b_{1}\right)+b_{2}$ (note the max term is gone).
    - This simplifies to $f(x) = \left(W_{1} W_{2}\right) x+\left(W_{2} b_{1}+b_{2}\right)$
    - Which is just $f(x) = \left(W_{3}\right) x+\left(W_{2} b_{1}+b_{2}\right)$ and the two weight matrices collapses into one.

> [!note]
> ReLU is a good default choice for most situations.

![[activation-function-comparison.png]]