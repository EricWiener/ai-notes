---
summary: Changes the negative scaling from Leaky ReLU from a constant to a learnable value
---

This gets rid of the additional hyperparameter for the from Leaky ReLU and makes the constant scalar for the negative side of the ReLU a learnable value.

$$
f(x) = \max(\alpha x, x)
$$

You back propagate into $\alpha$ and perform gradient descent on it. It could be a single value, or a vector of values the same size as $x$.

This is not as common to use as Leaky ReLU