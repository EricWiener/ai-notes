---
tags:
  - flashcards
source: https://en.wikipedia.org/wiki/Perplexity
summary: In NLP, perplexity is a way of evaluating language models. Models with a lower perplexity are better and will be more confident about their predictions.
publish: true
---
In natural language processing, **perplexity** is a way of evaluating language models. Models with a lower perplexity for test examples (drawn from the same distribution as the training examples) are ==better== and will be ==more confident== about their predictions. Perplexity tells you how confident the model was about the sequence is predicted.
<!--SR:!2028-09-12,1856,350!2024-02-24,558,330-->

Perplexity is defined as:
$$\mathcal{P}=\exp \left(-\frac{1}{n} \sum_{i=1}^n \log f_\theta\left(x_i \mid x_1, \ldots, x_{i-1}\right)\right)$$
If the perplexity is low, then the model is not very “surprised” by the sequence and has assigned on average a high probability to each subsequent token in the sequence.