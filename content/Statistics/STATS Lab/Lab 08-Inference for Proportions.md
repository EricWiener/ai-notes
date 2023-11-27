Files: lab08-slides-async.pdf

# Confidence Intervals

- Attempts to capture the fact that there is uncertainity in our point estimate (here $\hat{p}$ or $\hat{p}_1 - \hat{p}_2$).

$$
\text{point estimate} \pm \text{margin of error}
$$

- Centered at the point estimate
- Margin of error is 1/2 the width of the confidence interval (CI).

<aside>
💡 We have X amount of confidence that our true population proportion (or our true difference in population proportion) lies in a certain range.

</aside>

- We usually use a few standard errors as our margin of error.
    - margin of error = (a few) x (standard error of point estimate)
    

# Example: stats use of alchohol

In the Winter 2021 STATS 250 Student Survey, 1024 STATS 250 students out of 1533 reported occasional or daily use of alcohol.

## Conditions for confidence intervals for $p$

You need to make sure its appropriate to use the normal distribution.

1. Random sampling has occurred
2. We have a sufficiently large sample size, meaning ALL of the following are true:
    1. $n\hat{p}\geq 10$
    2. $n(1 - \hat{p}) \geq 10$

Note that you check the sample size condition using $\hat{p}$.

![[Lab 08 Inf/Screen_Shot.png]]

## **Computing a confidence interval**

![[Lab 08 Inf/Screen_Shot 1.png]]

You use the function `prop_test(x = # successes, n = sample size, conf.level = confidence level)`

![[Lab 08 Inf/Screen_Shot 2.png]]

- From the output, we really just care about the 95% confidence interval
- the $\hat{p}$ is the same as the one we calculated manually earlier (num who drank / sample size).

We are 95% confident that the true proportion of Stats 250 students who report occasional or daily use of alcohol is between 64.4% and 69.2%

# Hypothesis Test for $p$

- Previously we were did **Confidence Intervals for $p$**
- Now we are performing a hypothesis test to see whether stats 250 students consumed more alchohol than 12th graders.

![[Lab 08 Inf/Screen_Shot 3.png]]

## Conditions for Hypothesis Test for $p$

Two conditions:

1. Random sampling has occurred
2. We have a sufficiently large sample size, meaning ALL of the following are true:
    1. $np_0\geq 10$
    2. $n(1 - p_0) \geq 10$

<aside>
💡 Now we are checking sample size condition with $p_0$, the null hypothesis value. Previously, for the confidence intervals, we used $\hat{p}$.

</aside>

## Run hypothesis test

![[Lab 08 Inf/Screen_Shot 4.png]]

- We use `prop_test()` with `p = population value` and we set the `alternative = "greater"` to see the p-value of whether the value
- Z test statistic: this is $\frac{\hat{p} - p_0}{\sqrt{p_0*(1-p_0)/n}}$
- We ignore the confidence interval. The upper-bound on the confidence interval is 1.0 because we only used a one-sided hypothesis test. It's obvious that at most 1.0 (aka 100%) of students will self-report alcohol use.
- The $p$ value is approximately 0.

**Conclusion:**

Since our p-value is (**small**), we have (**strong**) evidence against the null hypothesis. Thus we have (**strong**) evidence to support that the true proportion of STATS 250 students who self-report consuming alcohol is **greater than 0.553**. 

# Example: transfer students

We took a random sample of 200 STATS 250 students who responded to the Student Survey, which asked them to tell us whether they were a transfer student, as well as their residency status (in-state vs. out-of-state/international). Of the 200 sampled students, 32 transferred to U-M. 22 of those transfer students are in-state, while 10 of those transfer students were from out-of-state. Of the 200 sampled students, 168 did not transfer to U- M. 97 of those non-transfer students are from in-state, and 71 of those non-transfer students are from out-of-state.

**Let's create a confidence interval for this difference in proportions.**

![[Lab 08 Inf/Screen_Shot 5.png]]

- We are trying to find the difference between the true population proportions.
- It's not interesting to find the difference between sample proportions because you can do this just by doing $\hat{p}_1 - \hat{p}_2$ as below.

![[Lab 08 Inf/Screen_Shot 6.png]]

## Conditions for confidence intervals for $p_1 - p_2$

![[Lab 08 Inf/Screen_Shot 7.png]]

![[Lab 08 Inf/Screen_Shot 8.png]]

- Note that you now need to check the conditions for both populations.

## Compute a confidence interval

![[Lab 08 Inf/Screen_Shot 9.png]]

- Here we pass in a vector for `x` and for `n`.
- The first entry in `x` is the number of successes in the first sample. The first entry in `n` is the sample size of the first sample. Similarly for the second sample. **Order matters**

![[Lab 08 Inf/Screen_Shot 10.png]]

- It calculates $\hat{p}_1$ and $\hat{p}_2$ for you (same as the values we calculated)
- It will always compute group 1 - group 2
- It also tells you the 99 percent confidence interval:

The 99% confidence interval for the true difference in proportions of in-state students for the two groups, transfer students and non-transfer students, is **(-0.123, 0.343)**.

# Example: transfers hypothesis test

We just tried to find a confidence interval for the true difference in proportions. Now, we can do a hypothesis test to see if the two proportions are equal or not (**hypothesis test for the difference in these proportions**)

### Setup question

![[Lab 08 Inf/Screen_Shot 11.png]]

### Conditions for $p_1 - p_2$

![[Exam 2 Rev/Screen_Shot 11.png]]

- Here, we use the **pooled estimate of the sample proportion** as the proportion when checking if the samples are large enough.
- You just add up total successes and divide by total sample size.
- We run a hypothesis test assuming $H_0$ is true until we can't assume it anymore, so we check conditions assuming $H_0$ is true.
    - $H_0$ says $p_1 = p_2$. The best we can do is get a single $\hat{p}$ that combines data from both groups.

![[Lab 08 Inf/Screen_Shot 12.png]]

- You now check with `pHatPooled`

### Run hypothesis test

![[Lab 08 Inf/Screen_Shot 13.png]]

- This p-value tells us likelihood $p_1 - p_2 > 0$ (note that we use the population proportions here - not the sample proportions).
- Ignore confidence interval here (we did a one-sided test but it should have been two-sided).