Created: July 15, 2020 7:52 AM
Type: Notes

**Gaussian Models** can be described entirely by their mean and variance. 

- Gaussian and normal models are the same thing.
- We assume data is coming from the reals (for the Bernoulli case we only considered data from a binary variable).

![https://paper-attachments.dropbox.com/s_1B0611DAAD9036AFC43296683495D7848754213B8F22EC19969E8C17BBE7EC02_1582573945832_Screen+Shot+2020-02-24+at+2.52.23+PM.png](https://paper-attachments.dropbox.com/s_1B0611DAAD9036AFC43296683495D7848754213B8F22EC19969E8C17BBE7EC02_1582573945832_Screen+Shot+2020-02-24+at+2.52.23+PM.png)

![https://paper-attachments.dropbox.com/s_BD7F2FDE7E65B3088F4C42F4FE8828E0217D091694E767B99FA988B443AE0080_1579806851955_Screen+Shot+2020-01-23+at+2.14.02+PM.png](https://paper-attachments.dropbox.com/s_BD7F2FDE7E65B3088F4C42F4FE8828E0217D091694E767B99FA988B443AE0080_1579806851955_Screen+Shot+2020-01-23+at+2.14.02+PM.png)

These graphs are like the marginal distribution charts we’ve seen, but continuous versions. You can take the marginal distribution to see the distribution of a single variable, or you can freeze one variable and take the conditional probability.

If variance of height tells you nothing about variance of weight (and vice-versa), their off-diagonal co-variances are 0.

**Parameterizing a distribution**: representing a model in terms of its parameters. We can parameterize the Gaussian distribution by mean and covariance.

# 1-Dimensional Gaussian
**We can model probability using Gaussian Model**

$P(X=x|\mu, \sigma)$
- X=x: observed data
- $\mu, \sigma$: unknown parameters

![https://paper-attachments.dropbox.com/s_1B0611DAAD9036AFC43296683495D7848754213B8F22EC19969E8C17BBE7EC02_1582574292038_image.png](https://paper-attachments.dropbox.com/s_1B0611DAAD9036AFC43296683495D7848754213B8F22EC19969E8C17BBE7EC02_1582574292038_image.png)

![https://paper-attachments.dropbox.com/s_285F14A6D23AA42271767F066E03A1EFD35060EE5D60C1D152305DE023DE01F3_1586177805180_Screen+Shot+2020-04-06+at+8.56.41+AM.png](https://paper-attachments.dropbox.com/s_285F14A6D23AA42271767F066E03A1EFD35060EE5D60C1D152305DE023DE01F3_1586177805180_Screen+Shot+2020-04-06+at+8.56.41+AM.png)

The **normal distribution**: $N(x | \mu, \sigma^2) \propto e^{-\frac{1}{2\sigma^2}(x-\mu)^2}$
- This is a probability density function, which means when you integrate it, it must add up to 1.
- Parameterized by the mean and the variance
- Nice properties: product of gaussian distributions forms gaussian
- **Central Limit Theorem:** expectation of the mean of any random variables (with zero-mean but any conditional density functions) converges to Gaussian
- ↳ good choice when dealing with noise and uncertainly
- Reducing data to Gaussian saves memory
- p(x) is the probability that x is true according to the model
- Mean value determines center of the distribution. Variance determines spread. The larger the variance, the more uncertain you are about the state.
- Unimodal and symmetric

Full Equation for a Gaussian Distribution:

$$
N(x | \mu, \sigma^2) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2\sigma^2}(x-\mu)^2}
$$

- The $\sigma^2$ term in the denominator of the exponent controls how fat or skinny (aka tall or short) the distribution is
- The $\mu$ in $(x - \mu)^2$ decides how shifted along the x-axis the distribution is
- $\frac{1}{\sigma \sqrt{2\pi}}$ is just a normalizing term to make sure the distribution adds up to 1.

### Characteristics:
1. Normal curves are symmetric
2. Normal curves are unimodal
3. Normal curves are bell-shaped
4. Normal curves are centered at the mean of the distribution
5. The total area under the curve is 1 (it represents a probability distribution)

# Merging two Gaussian Distributions
Let _X_ and _Y_ be independent random variables that are normally distributed, then their sum is also normally distributed. i.e., if
$$\begin{aligned}
& X \sim N\left(\mu_X, \sigma_X^2\right) \\
& Y \sim N\left(\mu_Y, \sigma_Y^2\right) \\
& Z=X+Y,
\end{aligned}$$
then
$$Z \sim N\left(\mu_X+\mu_Y, \sigma_X^2+\sigma_Y^2\right)$$
This means that the sum of two independent normally distributed random variables is normal, with its mean being the sum of the two means, and its variance being the sum of the two variances. The standard deviation is the square root of the variance $\sigma$ is the standard deviation and $\sigma^2$ is the variance. This means the square of the merged standard deviation is the sum of the squares of the standard deviations.

In order for this result to hold, the assumption that _X_ and _Y_ are independent cannot be dropped, although it can be weakened to the assumption that _X_ and _Y_ are [jointly](https://en.wikipedia.org/wiki/Multivariate_normal_distribution "Multivariate normal distribution"), rather than separately, normally distributed.

# MLE for univariate Gaussian
Given $S = \{\bar{x}\}_{i=1}^n$ drawn iid, we want to maximize the probability of $p_r(S_n) = \Pi_{i=1}^n p(\bar{x}^{(i)})$.

- We want to maximize $p(S_n)$ with respect to $\mu$
    - $\mu_{\text{MLE}} = \sum_{i=1}^n \frac{x^{(i)}}{n}$
    - If you want to figure out where to center your curve, you just need to add up your data points and take the average
- And with respect to $\sigma^2$
    - $\sigma^2_{\text{MLE}} = \sum_{i=1}^n \frac{(x^{(i)} - \mu_{\text{MLE}})^2}{n}$
    - The variance of the data we have

You can do this by taking the partial derivative.

