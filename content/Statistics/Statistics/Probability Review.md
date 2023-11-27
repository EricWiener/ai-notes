Created: July 15, 2020 7:49 AM
Type: Notes

[[Probabilit%20c01b8 Probability.pdf]]

## Marginal Probability

This goes from a joint distribution (three variables) to something smaller.

**Marginalization** is the process of getting an unconditional probability from the joint probability: $P(Y) = \sum_{z \in Z} P(Y, z)$

## Independence

$P(a \land b) = P(a)P(b)$ is they are independent from each other

Therefore:

$P(a |b) = \frac{P(a \land b)}{P(b)} = \frac{P(a)P(b)}{P(b)} = P(a)$

**Mutually exclusive**: two events can’t happen at the same time

**Independenc**e: knowing one event, doesn’t tell you anything about another

## Conditional Indepence

$P(X, Y|Z) = P(X|Z)P(Y|Z)$

This is $P((X\land Y | Z)$

If you observe z, then the two variables have no relation to each other.

Only independent when Z is true

Sometimes your sample space could be unrepresentative of your population. This makes it hard to sometimes find probability of an event (# observations for event A) / (total number of observations).

If the events aren’t mutually exclusive, don’t add them.

## Axioms of Probability

1. $0 \leq P(a) \leq 1$
    1. Probability of something happening is between 0 and 1
2. $P(\Omega) = 1$:
    1. $\Omega$ is all the possible outcomes
    2. Probability of every outcome added together is 1
3. $P(a \lor b) = P(a) + P(b)$
    1. Probability of either event happening is the sum of their probabilities if a and b are disjoint.
        1. **Disjoint:** can never occur simultaneously.
    2. $P(a \lor b) = P(a) + P(b) - P(a \land b)$ if not disjoint

## Types of Probabilities

P(X=x) - unconditional or **prior probabilities.** Refer to degrees of belief in the absence of other information.

**Conditional Probability**

$P(a|b) = \frac{P(a,b)}{P(b)}$. Tricky to get $P(b)$, since it’s hard to calculate for large spaces.

Ex: P(cat|the), very hard to get P(the) because you need corpus that represents all of English language

If A contains B and B occurred, then P(A) = 1.

Also $P(a, b) = P(a|b)P(b)$ - **product rule**

**Independence**

$P(a \land b) = P(a)P(b)$ if independent

$P(a) = P(a|b)$ - knowing b tells us nothing about a.

## Baye’s Theorem

Baye’s Theorem

$P(a \land b) = P(a|b)P(b)$

$P(b \land a) = P(b|a)P(a)$

$P(b|a)P(a)=P(a|b)P(b)$

$P(b|a) = \frac{P(a|b)P(b)}{P(a)}$

## Expectation

**Expected Value:**

$E = \sum_{k=1}^n x_k P(X=x_k)$ - weighted average according to probability

- $\mu = E$ - the mean of the distribution

**Properties:**

- $E[\alpha] = \alpha$
- $E[\alpha x] = \alpha E$
- $E[\alpha + x] = \alpha + E$
- $E[x + y] = E + E[y]$

**Variance**

How much you can expect variable to vary around its average value. The average of the squared distance from the mean. $\sigma^2 =E[(x-E)^2]$

- $\sigma$ is the standard deviation
- $\sigma^2$ is the variance

![https://paper-attachments.dropbox.com/s_1B0611DAAD9036AFC43296683495D7848754213B8F22EC19969E8C17BBE7EC02_1582573791074_Screen+Shot+2020-02-24+at+2.49.48+PM.png](https://paper-attachments.dropbox.com/s_1B0611DAAD9036AFC43296683495D7848754213B8F22EC19969E8C17BBE7EC02_1582573791074_Screen+Shot+2020-02-24+at+2.49.48+PM.png)