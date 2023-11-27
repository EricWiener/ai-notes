Files: mean_3.pdf
Tags: April 5, 2021 7:37 AM

- Two data sets are paired if each observations in one set has a special correspondence or connection with exactly one observation in the other data set
- Paired data have two features:
    - Each pair of data is associated with one observational unit
    - There is a built-in comparison: one number in each pair is for Condition A, the other for condition B
- Example: a teacher has students take an exam at the start of the year and end of the year to measure improvement. The first and final exam scores for each student form a pair.

### Comparing Means vs Paired Data

- In the example for a initial and final exam, an alternative approach is to obtain independent random samples from all the pretest and posttest measurements and compare the means.
- However, by examining the differences between the pretest and posttest measurements f or paired data, we eliminate the variability across different students and focus in on that improvement measure (posttest – pretest) itself.
- In general, when we study the difference between population means, studies leading to paired data are often more efficient—better able to detect differences between conditions—because we control for a source of variation (i.e., individual variability among the observational units).

## Notation

![[Paired Dat/Screen_Shot.png]]

- We have two measurements on each pair, so we can denote our data as above.
- There is dependence between the items in each pair because they came from the same observational unit.
- Instead of being interested in the two measurements separately, we are interested in the differences within each pair:

![[Paired Dat/Screen_Shot 1.png]]

- We then take these differences as our data and apply our **one-sample methods for inference.**
- We change our notation to indicate that we are working with differences. The parameter of interest is $\mu_d$, the population mean of the differences. The point estimate for $\mu_d$ is $\bar{x}_d = \frac{1}{n} (x_{1,2} + x_{2,d} + · · · + x_{n,d})$, the sample mean of the differences.

## Conditions for the sample mean of differences $\bar{x}_d$

The sampling distribution of $\bar{x}_d$ will be approximately normal when the

following two conditions are met:

1. Independence: The observations within the sample of differences are
independent (e.g., the differences are a random sample from the
population of differences).
2. Nearly normal: The distribution of the **population** of differences is nearly
normal or the sample sizes n is large enough.
    - We don't need the distributions of both populations (ex. before and after test scores) to be nearly normal. You just need the differences to be.

<aside>
💡 Note: **We do not need to check for independence between the samples**
because we already know that the samples are not independent.

</aside>

## Math

- The mean of this normal distribution is $\mu_d$
- The standard error of this distribution is $\text{SE}_{\bar{x}_s} = \frac{\sigma_d}{\sqrt{n}}$
    - $\sigma_d$ is the population standard deviation of the differences
    - $n$ is the number of pairs
- We usually don't know the population standard deviation $\sigma_d$, so we estimate it by the sample standard deviation $s_d$, which is computed based on the sample differences ($x_{1,2}, x_{2,d}, · · · , x_{n,d}$)
- We can then standardize $\bar{x}_d$ by subtracting the mean and dividing by the standard error in order to get a $t$ distribution with $n - 1$ degrees of freedom:
    
    $$
    t = \frac{\bar{x}_d - \mu_d}{s_d / \sqrt{n}}
    $$
    

## Hypothesis tests for paired data

![[Paired Dat/Screen_Shot 2.png]]

$$
t = \frac{\bar{x}_d - \mu_0}{s_d / \sqrt{n}}
$$

### Hypothesis Test Example

![[Paired Dat/Screen_Shot 3.png]]

![[Paired Dat/Screen_Shot 4.png]]

![[Paired Dat/Screen_Shot 5.png]]

![[Paired Dat/Screen_Shot 6.png]]

![[Paired Dat/Screen_Shot 7.png]]

![[Paired Dat/Screen_Shot 8.png]]

**Evaluate the p-value and the compatibility of the null model.**

Because the p-value is large (greater than 0.10), there is little evidence against the null hypothesis.

**Make a conclusion in the context of the problem. To what population can we
generalize these results?** 

The size of the bowl does not appear to influence the number of candies
taken, on average. We need to restrict these conclusions to UIUC students attending study sessions.

**Can we make a causal conclusion here?**

This was a randomized experiment, and the order in which each participant received a small bowl or a large bowl was randomly assigned. If we find a statistically significant difference we will be able to draw a cause-and-effect conclusion between the size of the bowl and the number of M&M’s taken.

## Confidence Interval for paired data

![[Paired Dat/Screen_Shot 9.png]]

$$
\bar{x}_d \pm t^* \frac{s_d}{\sqrt{n}}
$$