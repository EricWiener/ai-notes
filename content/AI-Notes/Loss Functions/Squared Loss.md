---
summary: permits small discrepancies, but penalizes large ones
---

$\text{Loss}(z) = \frac{z^2}{2}$. The squared term permits small discrepancies, but penalizes large ones. $z$ is the difference between our predicted value and the actual one. We divide by two because when we take the derivative later, the $2$ will cancel out.

> [!note]
> Least squares approximation is just the term for when you use squared loss to train

If you have the **raw predicted score** by the model $\hat{y} = f_{\theta}(x_i)$

$$
L(\hat{y}, y) = \sum_{k=1}^K (y_k - \hat{y}_k)^2
$$

This is easy to optimize and lets us use gradient descent, but it isn't great.