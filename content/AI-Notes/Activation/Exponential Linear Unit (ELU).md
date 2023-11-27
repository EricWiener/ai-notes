---
summary: All the benefits of Leaky ReLU, but more zero-centered and has gradient at zero. Requires exp() and scaling hyperparameter.
---
![[elu-graph.png.png]]
![[elu-math.png.png]]

This is an improvement on Leaky ReLU. It has all the benefits of ReLU, but it is more zero centered (it is shifted down a little to achieve this), and has a gradient at 0 (unlike ReLU which has a sharp non-differentiable point - ELU is smooth near zero middle).

Negative saturation (it heads to a certain value on the negative end) adds some robustness to noise compared to Leaky ReLU.

However, the computation requires `exp()` and also has a hyperparameter $\alpha$ you need to set.