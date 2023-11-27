---
tags: [flashcards]
source:
summary: The simplest optimization algorithm. Calculate the loss and gradient using the entire dataset.
---

The simplest optimization algorithm, but it works well in practice. Iteratively take small steps in the direction of the local steepest descent.

![[AI-Notes/Update Rules/batch-gradient-descent-srcs/Screen_Shot.png|400]]
![[AI-Notes/Update Rules/batch-gradient-descent-srcs/Screen_Shot 1.png|400]]

**Hyperparameter:**
- Weight initialization method
- Number of steps (generally more steps is better)
- Learning rate (how small of steps to take) - usually most important for deep learning models.

Usually gradient descent isn't a straight line to the global minimum.

Computing the **loss is very expensive** because you need to calculate the loss for even point and sum them. 

We subtract the gradient from the weights because if the gradient of the loss with respect to the weights is positive, that means moving in that direction will increase the loss. We want to decrease the loss, so we go opposite the gradient.