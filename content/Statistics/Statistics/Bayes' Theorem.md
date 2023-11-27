---
tags: [flashcards]
aliases: [Bayes' Rule, Conditional Probability]
source: https://www.investopedia.com/terms/p/posterior-probability.asp
summary:
---

### Bayes' Theorem Overview
Bayes' Theorem can be used to calculate the **posterior probability** of an event (the probability that event $A$ ocurrs given that event $B$ has ocurred).

Bayes' Theorem states:
$$P(A \mid B)=\frac{P(A \cap B)}{P(B)}=\frac{P(A) \times P(B \mid A)}{P(B)}$$
where:
- $A, B$ are events
- $P(B|A)$ is the probability of $B$ ocurring given that $A$ is true (likelihood).
- $P(A)$ is the probability of $A$ ocurring (prior)
- $P(B)$ is the probability of $B$ ocurring (evidence)
- $P(A|B)$ is the posterior probability 

### Probability Review
$0 \leq P(X) \leq 1$ aka $\sum P(X_i) = 1$ - the sum of all probabilities add up to 1

![[Definition of Conditional Probability]]

### Mathematical Explanation of Bayes' Theorem
$P(a \cap b) = P(a|b)P(b)$ (definition of conditional probability)

$P(a \cap b) = P(b|a)P(a)$ (definition of conditional probability)

$P(b|a)P(a) = P(a|b)P(b)$ (substitution)

$P(b|a) = \frac{P(a|b)P(b)}{P(a)}$ (divide both sides by $P(a)$)

### Example
You are worried you caught a dangerous disease, Hypothesitis. You look online and see that the likelihood of you having the disease given your symptoms is 0.95. However, this doesn't tell you the probability you have the disease.

- $P(H|E)$ is the probability you have the disease, Hypothesitis, given your symptoms.
- $P(E|H)$ is the probability you have the conditions you have given you have Hypothesitis.
- $P(E)$ is the probability you have the symptoms of Hypothesitis
- $P(H)$ is the probability of anyone having Hypothesitis

We can solve $P(H|E) = \frac{P(E|H)P(H)}{P(E)}$ to find the probability that you have Hypothesitis.

- $P(H)$: it turns out the disease is rare, so the probability of anyone having Hypothesitis is 0.0001
- $P(E|H)$: 0.95
- $P(E)$: the symptoms are having a runny noise, so this is 1/100 = 0.01

The probability that you actually have Hypothesitis are $P(H|E) = \frac{0.95 * 0.0001}{0.01} = 0.00095$, so the likelihood is actually very low.