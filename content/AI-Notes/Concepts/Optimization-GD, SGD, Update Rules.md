# Optimization: GD, SGD, Update Rules

Tags: EECS 498 F20

Finding a good value for $w$ means finding the lowest value for $w$.

$$
w* = \argmin_w L(w)
$$

![[AI-Notes/Concepts/optimization-gd-sgd-update-rules-srcs/Screen_Shot.png]]

When trying to find the minimum loss, we can't see which way to go in the valley. 

### Idea #1: Random Search

This is a bad idea. We just try random weight matrices and evaluate our loss. Choose the best result.

### Idea #2: Follow the slope

Even though we can't see off into the distance, we check where the gradient is sloping down.

We can use the derivative of a function to get its slope.

![[AI-Notes/Concepts/optimization-gd-sgd-update-rules-srcs/Screen_Shot 1.png]]

In multiple dimensions, the **gradient** is the vector of (partial derivatives) along each dimension. The slope in any direction of the **dot product** of the direction with the gradient. The direction of steepest descent is the smallest (most negative) **negative gradient**.

![[Using this definition of the gradient, you would need to slightly change each dimension individually to approximate the gradient at each dimension.]]

Using this definition of the gradient, you would need to slightly change each dimension individually to approximate the gradient at each dimension.

This isn't a good way to do it because this is just an approximation for the gradient and has to be done O(n) where n is the number of dimensions of W.

### Loss is a function of $W$.

Use calculus to compute an analytic gradient. It's much faster to calculate, but it is also error-prone.

**In practice:** always use **analytic gradient,** but use **numeric gradient** to check it (when deriving an analytic gradient yourself). This is called a **gradient check.** 

Sometimes the PyTorch automatically calculated derivatives aren't very fast. Sometimes it's better to calculate an analytic gradient yourself for very complicated stuff (ex. differentiable rendering).

# Batch Gradient Descent

The simplest optimization algorithm, but it works well in practice. Iteratively take small steps in the direction of the local steepest descent.

![[Batch Grad/Screen_Shot.png]]

![[Batch Grad/Screen_Shot 1.png]]

**Hyperparameter:**

- Weight initialization method
- Number of steps (generally more steps is better)
- Learning rate (how small of steps to take) - usually most important for deep learning models.

![[/Screenshot 2022-01-20 at 08.02.18@2x.png]]

Usually gradient descent isn't a straight line to the global minimum (the direction of the gradient is often not aligned with the direction of the global minimum).

Computing the l**oss is very expensive** because you need to calculate the loss for every point and sum them. 

We subtract the gradient from the weights because if the gradient of the loss with respect to the weights is positive, that means moving in that direction will increase the loss. We want to decrease the loss, so we go opposite the gradient.

# Stochastic Gradient Descent

Similar to Batch Gradient Descent, but we compute the loss and gradient with respect to a **minibatch** of the data (not the whole dataset). We now have the additional hyperparameter of **batch size** and **data sampling.** 

![[Stochastic/Screen_Shot.png]]

Use a batch size that is as large as possible. This is constrained by your GPU memory.

It's called stochastic gradient descent because we are putting a probabilistic interpretation of the data. The loss is the expected loss each time because we are only sampling from the full dataset.

![[Note that this is the expected loss, not the true loss]]

Note that this is the expected loss, not the true loss

![[Note that this is now the expected gradient (with respect to the expected loss).]]

Note that this is now the expected gradient (with respect to the expected loss).

 

### Problems

Issue #1: What if loss changes quickly in one direction and slowly in another? You will make slow progress and bounce back and forth.

![[This shows the gradient map of a 3D shape that looks like a taco shell]]

This shows the gradient map of a 3D shape that looks like a taco shell

Issue #2: In **local minimum** and **saddle points** you get a local minimum with zero gradient. You will get stuck there forever. Even if you don't get there exactly, the gradient will be very small and you will make very slow progress.

![[Stochastic/Screen_Shot 4.png]]

![[Stochastic/Screen_Shot 5.png]]

**Regularization** can have some nice benefits here because it adds a little bit of curvature everywhere in your objective landscape, so it can help you move along. However, if you consider the full loss function, it might still have local minimum that you can still get stuck in.

**Issue** #3: We also have an additional issues because of **SGD** because the gradients come from minibatches, so they can be noisy. 

# Update Rules

We can consider update rules to improve on SGD.

![[AI-Notes/Concepts/optimization-gd-sgd-update-rules-srcs/Screen_Shot 2.png]]

- First moment: this is what SGD + Momentum does. You accumulate velocity by adding up the gradients at each step. You add a decay term `rho` that will decrease the momentum, so it doesn't build up too much. If models that track first moments have `rho`.
- Second moments: this is what AdaGrad introduced. You accumulate the squared gradient over time. You then divide `dw` by this when updating weights. This has the affect of dampening down areas where you've seen a lot of gradient change and speeding up areas where there hasn't been a lot of gradient change.
- Leaky second moments: AdaGrad doesn't have a decay term built into it, so the second moment just builds up over time. RMRProp added a decay term and Adam also uses it.
- When you first start out with Adam, the `moment1` and `moment2` terms are quite small. You want to scale these up for the initial time-steps, so they behave normally.
- SGD is usually the slowest.
- SGD + Momentum usually overshoots and then comes back (but gives a large speed boost).
- RMSProp starts out fast and slows down towards the center.

```dataview
table summary as "Summary"
from  "AI-Notes/Update Rules"
```