---
tags: [flashcards]
source: https://en.wikipedia.org/wiki/Lp_space
summary: this is a generalization of the L0, L1, L2, ... norms.
---

For any real number $p \geq 1$, the $p$-norm $L^p$ is defined as:
$$\|x\|_p=\left(\left|x_1\right|^p+\left|x_2\right|^p+\cdots+\left|x_n\right|^p\right)^{1 / p}$$
aka:
$$\|x\|_p=\sqrt[p]{(\left|x_1\right|^p+\left|x_2\right|^p+\cdots+\left|x_n\right|^p)}$$
This just means you raise each element in the vector to the $p$ power, sum up all the values, and then take the $p$th root.

See:
- [[L0 Norm]]
- [[Regularization|L1 Loss]]
- [[Regularization|L2 Loss]]