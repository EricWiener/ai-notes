---
tags: [flashcards]
source: https://cs231n.github.io/neural-networks-3/#sgd
summary: An update on SGD that usually results in better convergence rates on deep networks. It can be thought of as a ball rolling down a hill picking up more velocity, so it can speed through slight uphills.
---

[[Stochastic Gradient Descent (SGD)]]

![[AI-Notes/Update Rules/sgd+momentum-srcs/Screen_Shot.png]]

The **momentum update** is an update on [[Stochastic Gradient Descent (SGD)]] that usually results in better convergence rates on deep networks. It can be thought of as a ball rolling down a hill picking up more velocity, so it can speed through slight uphills.

It can be implemented with the following:

```python
# Momentum Update
v = mu * v - learning_rate * dw
# Update the weights
w += v
```

- Where `v` is the velocity (how fast the ball is moving). This is usually initialized to all zeroes. You keep track of this internally within the optimizer.
- `mu` is called the "momentum" and is a hyperparameter that acts as friction for the velocity (it really dampens the velocity - it’s name is pretty shitty). It is usually set to `0.9`. When running cross-validation, it is usually set to `[0.5, 0.9, 0.95, 0.99]`. If you set it to `0`, then this behaves the same as SGD. This exists to prevent `v` from growing out of control and `mu * v` will decrease all values in `v` before updating with the current direction of steepest descent.
- `learning_rate` is a hyperparameter that decides how much to update based on the gradient of the weights
- `w` is the weights we are updating

> [!note]
> With Momentum update, the parameter vector will build up velocity in any direction that consistently has the largest negative gradient.

SGD + Momentum can help solve all the issues of SGD. It can help us power through local minimum/saddle points. It can also help account for noise from minibatches. Finally, it can help smooth out the bounciness.

### Alternative Representations

Note that the following two forms of SGD are **equivalent**:

![[Screenshot_2022-01-21_at_08.21.572x.png]]