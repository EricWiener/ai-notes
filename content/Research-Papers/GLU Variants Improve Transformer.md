---
tags:
  - flashcards
source: 
summary: 
aliases:
  - SwiGLU
---
Introduces the SwiGLU activation function which helps transformers.
$$\operatorname{SwiGLU}(x, W, V, b, c, \beta)=\operatorname{Swish}_\beta(x W+b) \otimes(x V+c)$$