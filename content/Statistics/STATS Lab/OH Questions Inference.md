![[OH Questio/Untitled.png]]

- I said `C` because the p-value will be less than 0.10, so we reject at 0.10
- However, don't we usually reject when it's less than 0.05?

### How do confidence intervals relate to hypothesis tests

Because confidence intervals are two sided, they correspond to a two-sided test.

A 95% confidence interval corresponds to a two-sided hypothesis test with an
alpha of 5%. However, for a one-sided test with an alpha of 5%, the
corresponding confidence interval would be 90%.

## Lab 8 Q5

- Block out the confidence interval part when looking at the hypothesis test.
- z-statistic for Q5 is talking about the difference in proportions $\hat{p}_1 - \hat{p}_2$.
    - A value of -5.734 is telling us the z-score is 5.734 standard deviations below 0.

## Lab 8 Q6

- If we used a 2-sided confidence test, shouldn't our z-statistic be two-sided?
- Can we just use the N(0, 1) distribution

- -5.8 is a test statistic. It is on scale of p_1 - p_2.
    - Confidence is on the scale of percents when talking about the difference in proportions
- The $p$-value is defined as the probability of obtaining a value of the statistic at least as extreme as the observed statistic when the null hypothesis is true.

## Lab 8 Q7:

- Should q7 == q5 answer?
- Previously when doing hypothesis tests with simulations, we don't need to check conditions.
    - We aren't using normal approximation.
    - We only need to use normal approximation condition check when using the normal approximation.
    

A two sided hypothesis test corresponds to a confidence interval.

Confidence level does not impact p-value. Confidence level is (1 - alpha).

We never set a critical value for the hypothesis test.

<aside>
💡 A hypothesis test's p-value doesn't depend on the critical level you set.

</aside>

- `plot_norm` limits the plot to +- 3 standard deviations
- Sample proportion is always inside the confidence interval (100% chance of being in the interval).