---
tags: [flashcards]
aliases: [gelu, GELU]
source: 
summary: Works like a stochastic mask of the data where small values are more likely to be set to 0. Commonly used in transformers.
---
![[gelu-graph.png.png]]
![[gelu-math.png.png]]

Idea: works like a stochastic mask of the data. You have data coming in from previous layer and you want to randomly set some neuron values to 0. You want to keep large values the same and set small values to 0 (randomly).

- Multiply input by 0 or 1 at random. Large values are more likely to be multiplied by 1 and small values are more likely to be multiplied by 0 (data-dependent dropout).
- Under some assumptions, this GELU layer is approximated by $x \sigma(1.702 x)$ where $\sigma$ is sigmoid and $x$ is the input to the layer.
- Looks like ReLU that is both smooth and dips slightly negative before the origin.
- Induces some regularization to the model.
- Very common in Transformers (BERT, GPT, GPT-2, GPT-3)