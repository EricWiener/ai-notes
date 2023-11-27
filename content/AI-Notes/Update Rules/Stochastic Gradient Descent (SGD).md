---
tags: [flashcards]
source: https://cs231n.github.io/neural-networks-3/#sgd
summary: Compute the loss and gradient with respect to a minibatch of the data.
---
Similar to Batch Gradient Descent, but we compute the loss and gradient with respect to a **minibatch** of the data (not the whole dataset). We now have the additional hyperparameter of **batch size** and **data sampling.** 

![[AI-Notes/Update Rules/stochastic-gradient-descent-(sgd)-srcs/Screen_Shot.png|400]]

Use a batch size that is as large as possible. Used constrained by your GPU memory.

It's called stochastic gradient descent because we are putting a probabilistic element to the data. The loss is the expected loss each time because we are only sampling.

![[AI-Notes/Update Rules/stochastic-gradient-descent-(sgd)-srcs/Screen_Shot 1.png|400]]

![[AI-Notes/Update Rules/stochastic-gradient-descent-(sgd)-srcs/Screen_Shot 2.png|400]]

> [!note]
> The gradient points in the direction of increase, but we usually want to minimize a loss function, so we **subtract the gradient.**
> 

With a vector of parameters `x` and the gradient `dx` you perform:

```python
x -= learning_rate * dx
```

Where `learning_rate` is a hyperparameter (fixed constant). When evaluated on the full dataset with a low enough learning rate, it is guaranteed to make non-negative progress on the loss function

### Problems

Issue #1: What if loss changes quickly in one direction and slowly in another? You will make slow progress and bounce back and forth.

![[AI-Notes/Update Rules/stochastic-gradient-descent-(sgd)-srcs/Screen_Shot 3.png]]

Issue #2: In **local minimum** and **saddle points** you get a local minimum with zero gradient. You will get stuck there forever. Even if you don't get there exactly, the gradient will be very small and you will make very slow progress.

**Regularization** can have some nice benefits here because it adds a little bit of curvature everywhere in your objective landscape, so it can help you move along. However, if you consider the full loss function, it might still have local minimum that you can still get stuck in.

![[AI-Notes/Update Rules/stochastic-gradient-descent-(sgd)-srcs/Screen_Shot 4.png|300]] ![[AI-Notes/Update Rules/stochastic-gradient-descent-(sgd)-srcs/Screen_Shot 5.png|300]]

**Issue** #3: We also have an additional issues because of **SGD** because the gradients come from minibatches, so they can be noisy.