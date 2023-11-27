Files: normal_2-1.pdf
Tags: March 8, 2021 9:04 AM

- The CLT involves sampling from a **general** population. In one round of sampling, let's say the values are $x_1, ..., x_n$, and the mean is denoted by $\bar{x}$. If we do this for a large number of times, we will have a collection of sample means $\bar{x}$ and we can visualize the distribution by looking at the histogram.
- The CLT says the **distribution of the sample mean** can be approximated by the normal distribution.

![[Central Li/Screen_Shot.png]]

![[Central Li/Screen_Shot 1.png]]

![[Central Li/Screen_Shot 2.png]]

![[Central Li/Screen_Shot 3.png]]

- Even if the original population doesn't have a normal distribution, the distribution of the sample means will be normal.

## Guidelines regarding CLT

The distribution of the sample mean can be approximated well by the normal curve under the following conditions:

- the population distribution is symmetric
- the population variance is small
- the sample size is small

The first two conditions can be compensated by a large sample size (as we see in skewed graph above). Generally speaking, the distribution of the sample mean is approximately normal as long as $n \geq 30$.

## Example: Normal Approximation

![[Central Li/Screen_Shot 4.png]]

![[Central Li/Screen_Shot 5.png]]

**Conduct a normal simulation based hypothesis test**

![[Central Li/Screen_Shot 6.png]]

This is not unusual. Therefore, we conclude there is little evidence that the current rate of “boomeranging” changed from the 1997 level of 13%.

**Try again with normal approximation**

![[Central Li/Screen_Shot 7.png]]

The CLT-based p-value is a bit bigger than the approximate p-value, but we would make the same decision.