Created: March 29, 2021 1:26 PM

## Conditions

The sampling distribution of $\bar{x}_1 - \bar{x}_2$ will be approximately normal when the following two conditions are met:

1. **Independence within each sample**: The observations within each sample are independent (e.g., we have a random sample from each of the two populations).
2. **Independence between the samples**: The two samples are independent of one another such that observations in one sample tell us nothing about the observations in the other sample (and vice versa).
3. **Nearly normal**: The distributions of both populations are nearly normal or the sample sizes are both large enough.

![[Inference /Screen_Shot 1.png]]

### Example

![[Inference /Screen_Shot 2.png]]

![[Inference /Screen_Shot 3.png]]

<aside>
💡 Box plots don't tell us if a distribution is symmetric.

</aside>

![[Inference /Screen_Shot 4.png]]

- When calculating the df by hand for a difference of two means, you need to use the $\min(n_1 - 1, n_2 - 1)$
- We want 99% confidence interval, so we get $t^*$ for `qt(0.995, 74)` because 99% of the data will be between 0.005 and 0.995.
- With 99% confidence, we estimate that the tuition at private schools is on average, between $1131 and $9269 more than out-of-state tuition at public schools.

**Based on the interval computed in b, does there appear to be a difference in mean tuition at private universities and mean out-of-state tuition at public universities? How would you explain this to someone who has not taken a statistics course?**

The confidence interval is entirely above $0, which suggests that the mean tuition at private universities is higher than the mean out-of-state tuition at public universities.