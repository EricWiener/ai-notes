---
summary: NP-hard loss function to minimize.
aliases: [Squared Hinge Loss]
---
**Empirical risk with hinge loss:**

Minimizing using the 0-1 loss function is [NP-hard](https://www.youtube.com/watch?v=YX40hbAHx3s). We need a better loss function in order to solve this efficiently. We can drop the indicator function and instead use hinge loss to improve our situation.

$R_n(\bar{\theta}) = \frac{1}{n} \sum_{i=1}^{n} max\{ 1 - y^{(i)}(\bar{x} \cdot \bar{\theta}), 0 \}$

![[hinge-loss-graph.png]]

Hinge loss doesn’t want data points to be right on the border of being correctly classified. This is why the diagonal intercepts the X-axis at 1 instead of 0. If a point is just barely correctly classified, it will still push it to be more correctly classified. Unlike the 0-1 loss, where the slope was constant (0), hinge loss is a convex function. Convex functions are nice because we can use gradient descent to iteratively move to a lower error until we find the global minimum.

Note that when $\bar{\theta} \cdot \bar{x} = 0$, the loss function is not differentiable. We choose to push the point into the flat line case because it is easier to not update the point than to update it.

**Note:** the true class $y$ is supposed to be {+1, -1} instead of {0, 1}.

> [!note]
> A linear classifier that uses hinge loss is called a support vector machine (SVM).

# Squared Hinge Loss
![[squared-hinge-loss.png]]

