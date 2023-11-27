---
tags: [flashcards]
source: https://ermongroup.github.io/cs228-notes/inference/sampling/
summary: ancestral sampling is used when you want to sample data that is conditioned on other data. You first sample from a distribution that isn't conditioned on anything and then sample from subsequent distributions conditioned on this value.
---

A process of producing samples from a probabilistic model by first sampling variables which have no parents using their prior distributions, then sampling their child variables conditioned on these sampled values, then sampling the children’s child variables similarly and so on. [Source](https://www.mbmlbook.com/LearningSkills_Diagnosing_the_problem.html).

### StackExchange QA
**Q: Why is ancestral sampling used in autoregressive models?**

You can use ancestral sampling in all autoregressive models where the conditional probability distribution is known given the previous samples, e.g. $p(x_t \mid x_{t - 1}, \dots, x_1)$.

For example, consider a first-order Markov process:
$$x_1\rightarrow x_2\rightarrow \dots \rightarrow x_{t-1}\rightarrow x_t$$

The joint probability distribution would be $$p(x_1, \dots, x_t)=p(x_1)p(x_2 \mid x_1) \dots p(x_t \mid x_{t - 1})$$
Assume you know all the conditional probabilities/densities here. Then, you can generate an autoregressive process via first sampling $x_1$, then sampling $x_2$ from $p(x_2 \mid x_1)$, then $x_3$ from $p(x_3 \mid x_2)$, and so on, reaching up to $x_t$. You start sampling at the very ancestor and keep moving up.

[Source](https://stats.stackexchange.com/a/532102)