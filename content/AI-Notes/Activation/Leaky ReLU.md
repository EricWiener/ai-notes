---
summary: This is an improvement on ReLU that doesn’t saturate on either end (the negative end has a slight slope)
---

![[leaky-relu-graph.png.png]]

This is an improvement on ReLU that does not saturate at either end (the negative end has a slope). This makes it so it **will not "die".**

It also has the other nice properties of ReLU: computationally efficient and it converges faster than sigmoid or tanh.

$$
f(x) = \max(\alpha x, x)
$$

It multiplies the value by a small positive constant ($\alpha$) if it is negative (otherwise, it just acts as the identify function). The constant is a hyperparameter that you need to tune. It is often set to $\alpha = 0.1$

The more hyperparameters you need, the more cross-validation you need.