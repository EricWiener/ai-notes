---
summary: "The risk is 0 for correct classifications / 1 for incorrect"
---

**Empirical risk with 0-1 loss:**
$E_n(\bar{\theta}) = \frac{1}{n} \sum_{i=1}^{n} [[ y^{(i)} \neq sign(\bar{x} \cdot \bar{\theta})]]$ - this is the indicator function (0 if not true and 1 if true). 

For linear classifiers, this is equivalent to $E_n(\bar{\theta}) = \frac{1}{n} \sum_{i=1}^{n} [[ y^{(i)}(\bar{\theta}^{(k)}\cdot \bar{x}^{(i)}) \leq 0]]$.
![[0-1-loss-graph.png.png]]

Minimizing using the 0-1 loss function is [NP-hard](https://www.youtube.com/watch?v=YX40hbAHx3s). Not a very good loss function since you are either perfectly correct or wrong. Hard to tell how to make minor adjustments.