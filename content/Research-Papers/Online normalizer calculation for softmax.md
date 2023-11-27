---
tags:
  - flashcards
source: https://arxiv.org/abs/1805.02867
summary: introduces a way of calculating Softmax with fewer memory accesses
---
### Naive Softmax Implementation
$$y_i=\frac{e^{x_i}}{\sum_{j=1}^V e^{x_j}}
$$
![[screenshot 2023-10-14_09_20_51@2x.png]]

### Overflow Safe Implementation
Softmax is prone to integer overflow in the $d_j$ update since you are summing up lots of numbers. An overflow safe implementation subtracts the maximum value from each value:
$$y_i=\frac{e^{x_i-\max _{k=1}^V x_k}}{\sum_{j=1}^V e^{x_j-\max _{k=1}^V x_k}}$$

![[screenshot 2023-10-14_09_20_56@2x.png]]

### Faster Calculation
Algorithm 3 calculates both the maximum value m and the normalization term d in a single pass over input vector with negligible additional cost of two operations per vector element. It re- duces memory accesses from 4 down to 3 per vector element for the Softmax function evaluation.
![[screenshot 2023-10-14_09_22_24@2x.png]]

Essentially, the algorithm keeps the maximum value $m$ and the normalization term $d$ as it iterates over elements of the input array. At each iteration it needs to adjust the normalizer $d$ to the new maximum $m_j$ and only then add new value to the normalizer. 