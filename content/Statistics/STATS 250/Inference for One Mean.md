Files: Mean_1.pdf
Tags: March 15, 2021 10:01 AM

# Student's t-Distribution

![[A few t-distributions with different degrees of freedom]]

A few t-distributions with different degrees of freedom

- A family of distributions indexed by a parameter called degree of freedom (df), which can be any positive ineger
- A t-distribution has a bell shape that looks similar to a standard normal distribution
- t-distributions are flatter than the standard normal distribution, but will approach the standard normal as df increases. When df > 30, the t and standard normal distributions are almost indistinguishable.
- The total area under a distribution curve is 1, so the t-distribution's flatter shape means more values are likely to fall beyond two standard deviations from the mean than under the normal distribution.

## R Code

![[Inference /Screen_Shot.png]]

- We can use `pt(c, df)` instead of `pnorm`
- We can use `qt(p, df)` instead of `qnorm`

# The inference of one population mean

### Motivation

- This is what the original paper addressed
- Suppose you have a population with mean $\mu$ and population standard deviation $\sigma$ and we want to conduct inference on $\mu$ (find out what it is) based on a random sample $x_1, ..., x_n$.
- A good point estimate of $\mu$ is the sample mean $\bar{x}$. We then **want to find the sampling distribution of $\bar{x}$** because we need the sampling distribution if we want to conduct a hypothesis test of construct a confidence interval.

### Standard error of $\bar{x}$

- The standard error of $\bar{x}$ is $\text{SE}_{\bar{x}} = \sigma / \sqrt{n}$ (this is given as a fact and not explained in this course).
- A large standard deviation $\sigma$ corresponds to a larger standard error.
    - If the data are more variable, then we will be less certain of the location of the true mean, so the standard error should be bigger.
    - On the other hand, if the observations all fall very close together, then $\sigma$ is likely to be small, and the sample mean should be a more precise estimate of the true mean.
- A larger sample size corresponds to a smaller standard error.
    - We expect estimates to be more precise when we have more data, so the standard error $\text{SE}_{\bar{x}}$ should get smaller when n gets bigger.

## Sampling distribution of $\bar{x}$

![[Inference /Screen_Shot 1.png]]

We can standardize $\bar{x}$ similarly to how we have calculated a z-score (but now we get a t-score).

$$
 \frac{\bar{x} - \mu}{s / \sqrt{n}}
$$

- We don't know the population standard deviation $\sigma$, so we replace it with the sample standard deviation $\mu$.
- People had incorrectly thought that this t-score value followed a normal distribution. However, it turns out it doesn't.
- If the data $x_1, ..., x_n$ form a random sample drawn from a nearly normal distribution, then the standardized sample mean $\frac{\bar{x} - \mu}{s / \sqrt{n}}$ is distributed approximately as $t$ with $df = n - 1$

## Conditions to apply t-distribution for inference about a single mean

**Independence**: 

The random sample assumption ensures that the observations are independent of one another, which is what we really need. 

We also have independence if we know that the data come from an experiment where each subject was randomly assigned to a group and the subjects do not interact.

If the data were not collected in one of these two ways, we need to carefully check to the best of our ability that the observations were independent.

**Nearly normal**: 

The observations are from a population with a nearly normal distribution. The nearly normal condition is difficult to verify with small data sets. We should

i. take a look at a plot of the data for obvious departures from the normal model, usually in the form of prominent outliers, and

ii. consider whether any previous experiences alert us that the data may not be nearly normal.

When the **sample size is somewhat large, we can relax the nearly normal condition**. For example, moderate skew is acceptable when the sample size is about 30 or more, and strong skew is acceptable when the sample
size is about 60 or more.

# Hypothesis Tests for a Single Mean

- Hypotheses and conclusions apply to the population(s) represented by the sample(s)
- If the distribution of a quantitate variable is highly skewed, we should consider analyzing the median rather than the mean (not covered in this course).

## Hypothesis Test Refersher

This is just a refresher of information we already learned.

### Steps of hypothesis test are:

1. Determine appropriate null and alternative hypotheses.
2. Check the conditions for performing the test.
3. Calculate the test statistic and determine the p-value.
4. Evaluate the p-value and the compatibility of the null model.
5. Make a conclusion in the context of the problem.

### Forms of hypotheses

Remember that our hypotheses come in the form of two competing claims.

To test a particular value of a population proportion, we have the following
possible pairs of hypotheses:

![[Right-sided, two-sided, and left-sided]]

Right-sided, two-sided, and left-sided

What is this $\mu_0$ and where does it come from? This is the **hypothesized value of the population mean $\mu$ (it comes from the research problem)** that we will use to build the null model. We will then check to see if our sample results are compatible with the null model.

# Examples

![[Inference /Screen_Shot 3.png]]

![[Inference /Screen_Shot 4.png]]

![[Inference /Screen_Shot 5.png]]

- The data isn't normal, but we have a lot of observations, so we relax the conditions.

![[Inference /Screen_Shot 6.png]]

- Told sample mean is $\bar{x} = 13.71$. Sample standard deviation $s = 6.5$
- This is a two-tailed test, so we multiply the one-tailed result by two (a two-sided test will usually have twice the p-value of a one-sided test).

![[Inference /Screen_Shot 7.png]]

# Confidence intervals for a single mean

- Based on a sample of $n$ indepedent observations from a nearly normal distribution, a confidence interval for the population mean $\mu$ is:
    
    $$
    \bar{x} \pm t^* \frac{s}{\sqrt{n}}
    $$
    
- where $\bar{x}$ is the sample mean, $s$ is the sample standard deviation, and $t^*$ corresponds to the confidence level and degrees of freedom. In this case, we use $df = n - 1$.
- Typical choices for confidence levels are 90%, 95%, and 99%, but any value larger than 0 and smaller than 100 can be chosen.
- The confidence interval tells us how confident we can be that the interval we construct contains the true population mean.
- We select $t^*$ so that the percentage of the t-distribution between $-t^*$ and $t^*$ is equal to the confidence level we've chosen for the interval.

## Finding $t^*$

![[Inference /Screen_Shot 8.png]]

- For the hypothesis test, we calculate $t$ and then plot a T (note the capital T) distribution with df degrees of freedom. We then find the area under the T distribution that is beyond $t$.
- For the confidence interval, we calculate the area we want below a certain value (ex. 97.5% of the area), specify the T distribution using $df$, and then we are given $t^*$ which is the specific value of $t$ that we should use to capture this amount of area.
- The $t^*$ value for a 95% confidence interval is larger than a $z^*$ because it is flatter.

## Example:

![[Inference /Screen_Shot 9.png]]

![[Inference /Screen_Shot 10.png]]

![[Inference /Screen_Shot 11.png]]

- $df$ is sample size - 1
- You could also find $t^*$ with `qt(.975, 15, lower.tail = TRUE)`
    - `qt(.025, 15, lower.tail = FALSE)` is finding the value at the right side of the distribution that only has 2.5% of the area above it.