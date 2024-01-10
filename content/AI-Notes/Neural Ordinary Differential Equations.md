---
tags: [flashcards]
source:
summary:
---
[Great Video from Maziar Raissi](https://youtu.be/K1vjpTUni2c)
[MathWorks](https://www.mathworks.com/help/deeplearning/ug/dynamical-system-modeling-using-neural-ode.html)
[Neural ODEs through Jax](https://implicit-layers-tutorial.org/neural_odes/)

With [[Research-Papers/ResNet#Residual Networks|residual networks]] we add a skip connection. This looks like $x_{l+1} = x_l + F(x_l)$. In the diagram below there is an additional [[AI-Notes/Activation/ReLU|ReLU]] activation added but you can think of it as two steps:
- The residual connection: $x_{l+1} = x_l + F(x_l)$
- Apply ReLU: $x_{l+2} = \text{relu}(x_{l+1})$
![[AI-Notes/Concepts/cnn-architectures-srcs/Screen_Shot 6.png]]

If you change the index $l$ to $t$ and interpret your layers as time steps, then you go from $x_{l+1} = x_l + F(x_l)$ to $x_{t+1} = x_t + F(x_t)$. This is similar to discretizing an [[AI-Notes/Definitions/Ordinary Differential Equation|ODE]] with your timestep is 1.

$F(0)$ is the input $x_0$
$F(T)$ is the final output. You calculate this using a black-box differential equation solver. **This paper introduces how to back-propagate through an ODE solver.**

