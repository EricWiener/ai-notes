---
tags: [flashcards, eecs498-dl4cv]
source: 
aliases: [L1 Loss, L2 Loss]
summary: L1 and L2 functions can be used as both regularization functions and as loss functions themselves.
---

## Overfit

**Problem:** loss functions encourage good performance on training data, but we really care about test data.

A model is **overfit** when it performs too well on the training data, and has poor performance for unseen data.

## Regularization
![[loss-function-with-regularization-term.png]]
The left term is the data loss (how well the model predicts the true label for data in the training dataset). On the right is the regularization term to prevent the weights from becoming too overfit.

We can add a regularization term to prevent the model from fitting too well to the training set, so it can generalize better.

$\lambda$ is the trade-off between fitting too well and generalization.

![[loss-func-w-reg-term-color-annotated.png]]

**Common Regularizations:**

![[l1-and-l2-reg-equations.png]]

L2 and L1 regularization are often used. The trade-offs are dataset dependent. L1 has a sparsity constraint, which drives many of the entries to be identically zero. L2 will prefer them to be small, but non-zero (wants to spread them out).

> [!note]
> You basically always want to use a non-zero **l2_decay**. If it's too high, the network will be regularized very strongly. This might be a good idea if you have very few training data. If your training error is also very low (so your network is crushing the training set perfectly), you may want to increase this a bit to have better generalization. If your training error is very high (so the network is struggling to learn your data), you may want to try to decrease it. [Source](https://cs.stanford.edu/people/karpathy/convnetjs/docs.html)
> 

**Difference between L1 and L2 behavior as explained by gradient descent:**
![[plot-of-l1-and-l2-loss.png]]
This plot shows the gradient of L1 and L2

Notice in the above plot, the gradient of L1 is higher than the gradient of L2 as you approach the origin. During back-propagation, L1 will cause larger updates to the weight matrix to make values closer to 0 while the gradient from L2 will become increasingly smaller near the origin. 

**More Complex:**
- Dropout
- Batch normalization
- Cutout, Mixup, Stochastic Depth, etc.

# L1 vs L2 Regularization

[[K-Nearest Neighbors]]

![[l1-and-l2-distance-from-origin.png]]

- L1 regularization will encourage sparsity because all the weight values are added up, so smaller weights are just as penalized as large weights. L2 regularization squares the weight values, so weights that are close to 0, are penalized much less than larger weights. Therefore, L2 is more likely to make the weights small, but not necessarily zero, while L1 will reduce the weights to zero.
- Turning of L1 or L2 regularization will allow the model to fit better to the **training** data

# L2 Regularization vs Weight Decay

![[l2-reg-vs-weight-decay.png]]

Both L2 regularization and weight decay regularize the model by penalizing large values of the weight matrix. 

**L2 Regularization:**
- L2 regularization does this by adding a term to the loss function penalizing the norm of the matrix.
- This causes the gradient to include a term proportional to the weight matrix.
- The gradient is fed to the optimization algorithm to get the step direction, which is then applied to the weight matrix

**Weight Decay:**
- Weight decay doesn't add a term to the loss function.
- It adds a term proportional to the weight matrix to the step direction coming out of the optimizer

**Comparison:**
- In general, these aren't the same (you get different sequences of weight matrices) for generic optimization algorithms.
- L2 Regularization has the weights added onto the loss function. Weight decay adds on the weights to the step size.
- The optimizer can in principle perform arbitrary computation on the gradient in order to compute the step direction. Therefore, the step direction coming out of an optimizer might in principle modify or erase the component that comes from the L2 regularization term.
- However, weight decay bypasses the optimization algorithm, so each update to the matrix will shrink it by precisely $\alpha \lambda w_t$.
- They are the same (you get the same series of matrices) for SGD, SGD + Momentum, but they aren't the same for things with adaptive learning rates (ex. Adam).