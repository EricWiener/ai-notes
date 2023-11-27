---
tags: [flashcards]
source: https://cs231n.github.io/neural-networks-3/#sgd
summary: added element-wise scaling of the gradient based on history. Will dampen down areas where the gradient is high and speed up areas where the gradient is very flat.
---

![[AI-Notes/Update Rules/adagrad-srcs/Screen_Shot.png]]

Added element-wise scaling of the gradient based on the historical sum of squares in each dimension. "Per-parameter learning rates". We keep track of the sum of the squared value of the gradients. We can then divide the gradient by the square root of the squared gradients. This will **dampen down areas where the gradient is high and speed up areas where the gradient is very flat.** This is because dividing by large numbers will shrink the update, but dividing by small fractions will significantly increase it.

However, AdaGrad will always be increasing. This can be an issue because you are constantly slowing down. This has good theoretical properties (convex optimization), but it can be undesirable because you can get stuck.

The small `1e-7` term is added because when you first start out, `grad_squared` will be all zeroes and you don't want to divide by zero.

```python
# Assume the gradient dw and parameter vector w
cache += dw**2
w += - learning_rate * dw / (np.sqrt(cache) + eps)
```

- `cache` has the same size as the gradient and keeps track of the per-element sum of squared rgadients
- The smoothing term `eps` (usually set between 1e-4 and 1e-8) avoids dividing by zero.