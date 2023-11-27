---
tags: [flashcards]
source: https://en.wikipedia.org/wiki/Perplexity
summary: In NLP, perplexity is a way of evaluating language models. Models with a lower perplexity are better and will be more confident about their predictions.
---
In natural language processing, **perplexity** is a way of evaluating language models. Models with a lower perplexity are ==better== and will be ==more confident== about their predictions.
<!--SR:!2028-09-12,1856,350!2024-02-24,558,330-->

A language model is a probability distribution over entire sentences or texts.

A model of an unknown probability distribution _p_, may be proposed based on a training sample that was drawn from _p_. Given a proposed probability model _q_, one may evaluate _q_ by asking how well it predicts a separate test sample $x_1, x_2, \dots, x_N$ also drawn from _p_.

Better models _q_ of the unknown distribution _p_ will tend to assign higher probabilities _q_(_xi_) to the test events. Thus, they have lower perplexity: they are less surprised by the test sample.