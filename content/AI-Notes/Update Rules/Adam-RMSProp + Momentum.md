---
tags: [flashcards]
aliases: [adam]
source: https://cs231n.github.io/neural-networks-3/#sgd
summary: Combines the concept of velocity from Momentum with RMSProp scaling based on history. This is the Professor Justin Johnson's go-to first approach.
---

> [!note]
> Combines the concept of velocity from Momentum with RMSProp scaling based on history. This is the Professor Justin Johnson's go-to first approach.

### Simplified version (no bias correction):
![[Screenshot_2022-01-21_at_09.05.402x.png]]

You can see that now both the momentum `momentum1` is used (this is like momentum), as well as the sum of squares of the gradients `momentum2` (like AdaGrad/RMSProp).

**Simple Implementation (numpy)**
```python
m = beta1*m + (1-beta1)*dw
v = beta2*v + (1-beta2)*(dw**2)
w += -learning_rate * m / (np.sqrt(v) + eps)
```

This update looks very similar to RMSProp, except the smooth version of the gradient `m` is used instead of the raw (and possibly noisy) gradient vector `dw`. 

```python
# This is what RMSProp would look like using the same variable names
v = beta2*v + (1-beta2)*(dw**2)
# Only difference is that we are using dw below instead of the smooth version
# of the gradient (m)
w += -learning_rate *** dw** / (np.sqrt(v) + eps)
```
    
**Similarity to Momentum**
![[Screenshot_2022-01-21_at_09.04.152x.png]]
    
**Similarity to AdaGrad / RMSProp**
![[Screenshot_2022-01-21_at_09.04.252x.png]]

**Default Values:**
- `eps = 1e-8`
- `beta1 = 0.9`
- `beta2 = 0.999`
- `learning_rate = [1e-3, 5e-4, 1e-4]`

### Full Version:
When you first start out (t=0), `moment2` is very small. Since we are dividing `dw` by `moment2`, this can result in large steps in the first few iterations. The full version of Adam includes a **bias correction** mechanism to account for this.
![[AI-Notes/Update Rules/adam-rmsprop+momentum-srcs/Screen_Shot.png]]

When you first start out, you will be raising `beta1` and `beta2` to the power of a small number, but as time increases, `t` will be bigger. **This results in scaling up `moment1` and `moment2` more for the first few steps**. Because both betas are smaller than 1.0, they will decay with time quickly. At `t=0`, though, they will be their original values. 

- Since we do `(1 - beta1 ** t)`, at `t=0`, this will be like `(1 - 0.9) = 0.1`. Dividing by this small number will make the moment larger.
- When `t=5`, we will get `(1 - 0.9^5) = 0.40951`, so dividing by this won't make the moment that much bigger.

Note that now the update is a function of the iteration as well as the other parameters.

```python
# t is your iteration counter going from 1 to infinity
m = beta1*m + (1-beta1)*dw
mt = m / (1-beta1**t) # NEW
v = beta2*v + (1-beta2)*(dw**2)
vt = v / (1-beta2**t) # NEW
w += - learning_rate * mt / (np.sqrt(vt) + eps)
```