---
tags: [flashcards, eecs445]
source:
summary:
---

Tags: EECS 445

We now want to predict a continuous range of values instead of classifying values.

$y \in \mathbb{R}$

$f: \mathbb{R}^d \rightarrow \mathbb{R}$ where $f \in F$

![[AI-Notes/Concepts/Linear Reg/Screen_Shot.png]]

## Empirical Risk:
$R_n(\bar{\theta}) = \frac{1}{n}\sum_{i = 1}^n \text{Loss}(y^{(i)} - (\bar{\theta}\cdot\bar{x}^{(i)}))$ where $y^{(i)}$ is the actual value and $\bar{\theta}\cdot\bar{x}^{(i)}$ is the predicted value.

Previously we were using $\eta\nabla_{\bar{\theta}}\text{Loss}_h(y^{(i)}(\bar{x} \cdot \bar{\theta}))|_{\bar{\theta} = \bar{\theta}^k} = \eta\nabla_{\bar{\theta}}\max\{ 1 - y^{(i)}(\bar{x} \cdot \bar{\theta}), 0\}$ for **linear classification**. This made sense because we were using hinge loss, so if the value of $(\bar{x}\cdot\bar{\theta})$ was the same as $y^{(i)}$, then this value became positive and when taking the gradient of the hinge loss with stochastic gradient descent, the update for this point was 0. However, when these were opposite values, hinge loss would make us update for these points.

Now, $y^{(i)}$ can take on a continuous range of values. Therefore, we need to find the difference between the actual $y^{(i)}$ and the predicted value by $\bar{x} \cdot \bar{\theta}$. We are going to use the squared loss, so this difference will be handled the same whether it is positive or negative.

**Empirical Risk vs. Loss Function:**
- Loss Function: a loss or risk function, $\mathcal{L}(\hat{y}^i, y^i)$, quantifies how well (more accurately, how badly) $\hat{y}$ approximates y. smaller values of $\mathcal{L}(\hat{y}^i, y^i)$ indicate that $\hat{y}$ is a good approximation of $y$
- Empirical Risk: the empirical risk is the average loss over the data points. $\mathcal{L} = \frac{1}{n}\sum_{i=1}^n \mathcal{L}(\hat{y}^i, y^i)$

> [!note]
> Empirical Risk Minimization is the fancy term for trying to minimize empirical risk to train a machine learning model (supervised learning)
> 

## Squared Loss
$\text{Loss}(z) = \frac{z^2}{2}$. The squared term permits small discrepancies, but penalizes large ones. $z$ is the difference between our predicted value and the actual one. We divide by two because when we take the derivative later, the $2$ will cancel out.

## SGD for Linear Regression with Least Squares

$R_n(\bar{\theta}) = \frac{1}{n}\sum_{i = 1}^n \frac{(y^{(i)} - (\bar{\theta}\cdot\bar{x}^{(i)}))^2}{2}$ when using squared loss. When $y^{(i)}$ and $\bar{\theta}\cdot\bar{x}^{(i)}$ are the same value, the loss is zero. Otherwise, you square it and divide by 2.

$\nabla_{\bar{\theta}}\text{Loss}_h(y^{(i)}-(\bar{x} \cdot \bar{\theta})) = (y^{(i)}-(\bar{x} \cdot \bar{\theta}))\bar{x}$

**Algo:**

$k=0, \space \bar{\theta}^{(0)} = \bar{0}$ (initialize weights to zero)

while **convergence criteria** is not met:

- randomly shuffle points
- for i = 1 , … , n:
- $\bar{\theta}^{(k+1)} = \bar{\theta}^{(k)} + \eta\nabla_{\bar{\theta}}\text{Loss}_h(y^{(i)}-(\bar{x} \cdot \bar{\theta}))|_{\bar{\theta} = \bar{\theta}^k}$ # look at only ith data point
- k++

> [!note]
> We use the iterative approach to solving least squares when we have a lot of data and solving the closed form solution is too expensive (most of the time).
> 

## Closed Form Solution to Least Squares

![https://paper-attachments.dropbox.com/s_A1BD54ACBB709B053D07498FD73B8ACEC9152687B44C365C0681D91525E26B76_1582322059698_IMG_0DDB18A2AB3E-1.jpeg](https://paper-attachments.dropbox.com/s_A1BD54ACBB709B053D07498FD73B8ACEC9152687B44C365C0681D91525E26B76_1582322059698_IMG_0DDB18A2AB3E-1.jpeg)

> [!note]
> Note when implementing this on a computer, you don't want to calculate a matrix inverse. Instead you should solve for the linear system given by $b = A\theta^*$
> 

# Regularization

If $X^TX$ is singular, you won’t be able to take the inverse. This happens if the columns are linearly independent, which is caused by redundant features. You can solve this issue by removing redundant features with regularization.

**Objective Function:**

$J_{n, \lambda}(\bar{\theta}) = \lambda Z(\bar{\theta}) + R_n(\bar{\theta})$

- $Z(\bar{\theta})$ is the regularization term
- $\lambda$ is a hyperparameter that allows us to control the trade off

We will attempt to minimize the objective function.

**Regularization Term:**

- We want the regularization term to be convex and smooth. When adding two convex functions, you get a convex functions. This will allow us to take the gradient of the objective function.
- We want to force the components of $\bar{\theta}$ to be small (close to zero). This is because small variations in certain aspects of the feature vectors should not lead to a large variation in the predicted label.
- Popular choice is any L norm
- Usually we use $L_2 \space norm$

---

# Ridge Regression:

**L2 Regularization:** $Z(\bar{\theta}) = \frac{||\bar{\theta}||_2^2}{2}$.

Ridge regression uses L2 normalization with squared loss.

$J_{n, y}(\bar{\theta}) = \lambda \frac{||\bar{\theta}||^2}{2} + \frac{1}{n}\sum_{i=1}^n \frac{(y^{(i)}-(\bar{\theta}\cdot\bar{x}^{(i)}))^2}{2}$

- When $\lambda = 0$, this is linear regression with squared loss
- When $\lambda = \infty$, this is minimized at $\bar{\theta} = \bar{0}$.
- Want to balance these two extremes.

The **key difference** between L1 and L2 normalization is that **L1 Norm** shrinks the less important feature’s coefficient to zero thus, removing some feature altogether. So, this works well for **feature selection** in case we have a huge number of features.

## Closed Form Solution

![https://paper-attachments.dropbox.com/s_A4C825660E35A7755F7101E2BEDDC5BE1AD9F09DDE018AC214978A1AE9A87328_1582333338691_IMG_2D743387E602-1.jpeg](https://paper-attachments.dropbox.com/s_A4C825660E35A7755F7101E2BEDDC5BE1AD9F09DDE018AC214978A1AE9A87328_1582333338691_IMG_2D743387E602-1.jpeg)

The new closed form expression $(\lambda 'I + X^TX)^{-1}X^T\bar{y}$ as invertible as long as $\lambda > 0$

## Why closed form solution is always invertible

**Linear algebra:**

- Eigenvalues: if $A\bar{x} = v\bar{x}$ where $v$ is a scalar, then $(\bar{x}, v)$ is an eigenvector, eigenvalue pair of matrix A.
- All eigenvalues are positive in a positive definite matrix
- All eigenvalues are non-negative in a semi-definite matrix
- Positive definite matrices are invertible

**Applying the linear algebra:**

- $X^TX$ is a positive semi-definite matrix
- If matrix A has eigenvalue $k$, then $A + \lambda I$ has eigenvalue $k + \lambda$

Therefore, since $X^TX$ has eigenvalues $\geq$ 0, then adding a scalar greater than 0 to all of them, will boost all the eigenvalues up to be positive. This makes $\lambda 'I + X^TX$ invertible and solves our invertibility problem.

## Issues with Least Squares

- You can't output vertical lines (that aren't along y-axis) because we parameterized our model as $y = mx + b$ and then only way we could represent a vertical line was if $b = \pm \infty$

![[AI-Notes/Concepts/Linear Reg/Screen_Shot 1.png]]

- Not rotationally invariant: the line will change depending on orientation of points

## Alternate Formulation: Total Least Squares

We can rewrite the equation as $ax + by + c = 0$ where a point lies on the line only if $l^Tp = 0$ where $l = [a, b, c], \ \ \ p = [x, y, 1]$.

Since you can always just scale up $l$ and get the same results, we pick $a, b, d$ such that 

- $||n||_2^2 = ||[a, b]||_2^2 = 1$ ($n$ is a unit vector)
- $d = -c$.

This gives a 1-1 correspondence between the algebraic representation of the line and the geometric object it corresponds to. We can rewrite this as:

$$
ax + by - d = 0 \\ n^T[x, y] - d = 0
$$

**We can now compute the orthogonal distance between a point and the line:** 

$n$ is a unit vector and the distance from a point to line is given by:

$$
\frac{n^T[x, y] - d}{||n||_2^2} = n^T[x, y] - d
$$

![[AI-Notes/Concepts/Linear Reg/Screen_Shot 2.png]]

This is now rotationally invariant because we are measuring the orthogonal distance to the line. 

Previously we measured the vertical distances between the points and the lines.

![[AI-Notes/Concepts/Linear Reg/Screen_Shot 3.png]]

Now we have a rotational symmetry because we measure orthogonal distance. 

We can also represent a vertical line. 

> [!note]
> This formulation is called Total Least Squares and is good for images where points can be rotated and we need vertical lines (ex. looking at a column in an image).
> 

![[AI-Notes/Concepts/Linear Reg/Screen_Shot 4.png]]

## Least Squares vs. Total Least Squares

We usually use Least Squares in a machine learning context where we are trying to make predictions based on the value of the x-axis and want to minimize the error of the predictions. This is why Least Squares only measures distance along the vertical axis because we only care about how wrong our prediction was.

Total Least Squares is more like we have a bunch of points and want to find the line that goes through the points best. If we rotated the points, we would want to find a rotated version of the same line. This is more appropriate for computer vision where you detect a bunch of keypoints in an image and want to find the line that best passes through all the keypoints.

![[AI-Notes/Concepts/Linear Reg/Screen_Shot 5.png]]

Least Squares is very sensitive to outlier data. Because our objective function involves the squared error, the further away an outlier is, the more it will contribute to the error, so it has more weight than inlier data.

![[AI-Notes/Concepts/Linear Reg/Screen_Shot 6.png]]

![[AI-Notes/Concepts/Linear Reg/Screen_Shot 7.png]]

We can fix this by:

- Replacing the L2 error (sum of the squares of the errors) with L1 error (absolute value of the error), or Huber (quadratic near zero and then linear as you move away from zero).
- However, these fixes often have no closed form solution, typically not easy to implement, and sometimes not convex. They also don't work well for lots of outliers (sometimes in CV the majority of the data are outliers).