---
tags:
  - flashcards
source: 
summary: 
aliases:
  - Factorization
  - factorize
---
See [[CMU Factoring with conditionals Bayes rule.pdf]] for more info.

When using probabilities, you often describe the probability of multiple events happening at the same time. You can **factor** the probability distribution:
$$P(X)=F_{1}(X)F_{2}(X)\dots F_{n}(X)$$
Where $X$ stands for the list of all random variables or events you want to reason about and each $F_k(X)$ encodes some understandable part of the model.

To calculate the probability of $X$ taking on a specific value $x$, you just need to calculate the product of the probability of each factor taking on that value:
$$P(X=x)=F_{1}(x)F_{2}(x)\dots F_{n}(x)$$
If the events are independent, you can just multiply the probabilities of the individual events:
$$P(X_{1},X_{2})=P(X_{1})P(X_{2})
$$

If the events are conditionally independent (ex. events $X, Y$ are independent given you know event $Z$ has occurred or not, $P(X, Y \mid Z)=P(X \mid Z) P(Y \mid Z)$), you can break them as follows:
$$P(X,Y,Z)=P(Z)P(X,Y\mid Z)=P(Z)P(X\mid Z)P(Y\mid Z)
$$