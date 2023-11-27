---
summary: Improvement to Sigmoid (zero-centered), but still kills gradient
---
![[tanh-graph.png.png]]

This is basically a scaled and shifted version of sigmoid. It squashes numbers to range [-1, 1]. 

> [!note]
> It is zero centered, unlike sigmoid, but it also still kills gradients at the flat ends.
The derivative of `tanh` is $1 - \tanh^2(x)$. This can also be written as $z = \tanh(x)$ and the derivative is $1 - z^2$

The gradient for large magnitude `x` will be close to 0 since `tanh` has flat ends.