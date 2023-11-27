---
tags:
  - flashcards
source: ChatGPT
summary: represents the probability distribution of the uncertainty about certain values of a statistical model (ex. which cookie bowl did you select) given observed data (the type of cookie you got)
---

> [!PRIVATE] What is the posterior distribution?
> ??
> It represents the probability distribution of the uncertainty about certain values of a statistical model (ex. which cookie bowl did you select) given observed data (the type of cookie you got).

In statistics, the posterior distribution is a probability distribution that represents the uncertainty about the parameters of a statistical model after taking into account the observed data. It is named "posterior" because it is derived from the application of Bayes' theorem, which updates the prior beliefs about the parameters with the likelihood of the observed data.

Here's the basic idea:
1. **Prior Distribution (Prior):** This represents your initial beliefs or knowledge about the parameters before observing any data.
2. **Likelihood Function (Likelihood):** This represents the probability of observing the data given the parameters.
3. **Posterior Distribution (Posterior):** This is the updated distribution of the parameters after taking into account the observed data. It is obtained by applying Bayes' theorem, which combines the prior distribution and the likelihood function.

Mathematically, if $\theta$ represents the parameters of the model and D represents the observed data, Bayes' theorem is expressed as:

$$P(\theta|D)\equiv\ {\frac{P(D|\theta) \cdot P(\theta)}{P(D)}}$$

where:
- $P\left(\theta|D\right)$ is the posterior distribution.
- $P(D\vert\theta)$ is the likelihood of the data given the parameters.
- $P(\theta)$ is the prior distribution.
- $P(D)$ is the marginal likelihood or evidence, which is the probability of observing the data under all possible parameter values.

The posterior distribution is particularly useful in Bayesian statistics, as it provides a way to update beliefs about the parameters based on observed data. It can be used for parameter estimation, hypothesis testing, and model comparison. In practice, obtaining the exact form of the posterior distribution may be analytically challenging, and various computational methods, such as Markov Chain Monte Carlo (MCMC) or variational inference, are often employed to approximate it.

### Example
Suppose you have two bowls of cookies:
- Bowl 1: 30 vanilla cookies, 10 chocolate cookies
- Bowl 2: 20 vanilla cookies, 20 chocolate cookies

Now, you pick a bowl at random, and then pick a cookie from that bowl. You randomly select a cookie and find that it's a chocolate cookie. The question is: What is the probability that it came from Bowl 1?

Let's define some terms:
- $B_1$: Event that you chose Bowl 1.
- $B_2$: Event that you chose Bowl 2.
- $C$: Event that you picked a chocolate cookie.

We want to find $P(B_1 | C)$, the probability that you chose Bowl 1 given that you picked a chocolate cookie. We can use Bayes' theorem for this:
$$ P(B_1 | C) = \frac{P(C | B_1) \cdot P(B_1)}{P(C)} $$

- $P(C | B_1)$: Probability of picking a chocolate cookie given that Bowl 1 was chosen. In Bowl 1, this is $\frac{10}{40}$.
- $P(B_1)$: Probability of choosing Bowl 1 initially. Since you randomly picked a bowl, this is $\frac{1}{2}$.
- $P(C)$: Probability of picking a chocolate cookie, regardless of the bowl. This is given by the law of total probability:

$$ P(C) = P(C | B_1) \cdot P(B_1) + P(C | B_2) \cdot P(B_2) $$

Substitute these values into Bayes' theorem, and you get the posterior probability $P(B_1 | C)$.

In this example, the calculations would show that the probability of having picked Bowl 1 given that you picked a chocolate cookie is about $\frac{10}{30}$ or $\frac{1}{3}$. This illustrates how Bayes' theorem updates our probability beliefs based on new evidence (picking a chocolate cookie). The posterior distribution, in this case, represents the updated probability of each bowl being chosen.