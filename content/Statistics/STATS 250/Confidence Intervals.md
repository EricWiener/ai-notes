Files: Normal_4.pdf
Tags: March 8, 2021 9:03 AM

- One of our goals is to use sample statistics to estimate population parameters. A point estimate provides a single plausible value for a population parameter using collected sample data.
- Example: we use the sample mean $\bar{x}$ to estimate the population mean $\mu$ and we use the sample proportion $\hat{p}$ to estimate the population proportion $p$.
- Samples vary, so sample statistics may also vary.
- A point estimate gives us an estimate for a parameter, but it is useful to provide a margin of error for this estimate (a plausible range of values for the parameter based on the estimate).
- Statiscians call this range of plausible values a confidence interval.

## Sample Distributions

- Point estimates, such as the sample proportion or the difference between sample proportions, vary from sample to sample.
- Given an estimation problem, we can talk about the distribution and variabilityu of all possible estimates. **This distribution is called sampling distribution.**
- Ex: we can think about the margin of error of an estimate in terms of the standard deviation of the sample distribution.
    - Note: the standard deviation is generally unknown.

### Example

Consider the problem of estimation population proportion $p$ using sample proportions with sample size $n = 50$. Since, 

![[Confidence/Screen_Shot.png]]

- Above, we simulated with sampling with $p = 0.5$ and $p = 0.7$.
- The variabiities of the two sampling distributions are different because the sampling distribution depends on $n$ and $p$.
- Since we do not know $p$ (we are trying to estimate it), we can't use simulations to obtain the sampling distribution (because we need to know $p$ in order to simulate sampling from the population).

# Standard Error

- The standard error (SE) of a point estimate is the estimated standard deviation of the point estimate (or the sampling distribution).
- Example: Assuming this estimate is sufficiently accurate, if we had a point estimate with SE = 4.2 units, that would mean that the point estimate, over repeated samples, would be approximately 4.2 units away from the parameter it estimates, on average.
- The way we compute SE varies depending on the type of problem.

### Example: estimating the proportion of students who returned to A2

In a random sample of 320 undergraduate students in Stats 250, 251 returned to Ann Arbor for Fall 2020 classes. Thus, $\hat{p} = 251/320 \approx 0.784$. Using this information, estimate the proportion of all Stats 250 students who returned to Ann Arbor for Fall 2020 classes

- Suppose we were able to sample repeatedly (which we can't in real life).
- We would be able to obtain the sampling distribution.

The plot to the right depicts how 5000 sample proportions for this situation would lead to the sampling distribution. The plot is shaped like the normal curve and centered at the population proportion (which we do not know!).

![[Confidence/Screen_Shot 1.png]]

- The results of our sample proportion, $\hat{p} = 0.784$ is likely somewhere towards the center of the distribution (but not at the exact center).
- We can estimate the standard deviation of the distribution to be 0.023 (how to calculate this will be taught later).
    - Because this is just an estimate of the SD, we call is the **standard error**.

Now we can use this information to construct a plausible range of values for the population of Stats 250 students who returned to A2 for Fall 2020.

![[Confidence/Screen_Shot 2.png]]

- We decide to give an interval that is within 2 standard errors (**can't talk about standard deviations here)** of $\hat{p}$.
- Note that the $\hat{p}$ is the original sample proportion we got.

# Confidence Interval

## Interpretation of confidence inteval

Working off the above example, and assuming the CLT applies, we can approximate the sampling distribution of $\hat{p}$ by $N(p, \sigma)$ for some $\sigma$.

![[Confidence/Screen_Shot 3.png]]

- If $\hat{p}$ is within 2 standard deviations of the mean, which happens about 95% of the time (68-95-99.7 rule), then the interval $\hat{p} + 2\sigma$ will contain the proportion.
- Therefore, approximately 95% of the intervals we create using this method will include the parameter (since we are assuming $\hat{p}$ is close to the center.
- We call this a **95% confidence interval.**

## Confidence Interval Key Points

1. General format for a confidence interval is given by:
    
    $$
    \text{sample estimate} \pm \text{(a few) standard errors}
    $$
    
2. The value of the sample estimate will vary from one sample to the next. The values often vary around the population parameter, and the standard error gives an idea about how far the sample estimates tend to be from the true population proportion on average.
3. The standard error of the sample estimate provides an idea of how far away the estimate would tend to vary from the parameter value (on average).
4. The "few" or number of standard errors we go out each way from the sample estimate will depend on the coverage rate we pick (i.e. how confident we want to be). We call "(a few) standard errors" **the margin of error.**
5. The coverage rate of "how confident we want to be" is referred to as the confidence level. This level reflects our confidence in the **procedure** - how sure we feel that the interval contains the true parameter.
    - If the confidence level is 95%, then it means 95% of all the samples will lead to confidence intervals that include the true parameter and the other 5% won't.
    - In practice, we will only have one sample and one interval. It either contains the population parameter or it doesn't.

## Confidence Interval Big Idea

- 95% of the time (or whatever confidence you choose), a sampled value will be within 2 standard deviations of the true center of the distribution.
- Therefore, 95% of the sampled values you pick will have the mean within 2 standard deviations of it.
- Therefore, you can estimate the standard deviation with the standard error and say that the true center of the distribution is within 2 SE of the sampled value you picked with 95% confidence.

### Example: Pass or Fail

![[Confidence/Screen_Shot 4.png]]

![[Confidence/Screen_Shot 5.png]]

- We will talk about how to calculate SE later on.
- Here we use a range of 1.96 SE. This is the range you need to capture 95% confidence. Earlier we had just been approximating this to be 2.
- We can now say what range the actual difference between $p_{\text{urban}}$ and $p_{\text{rural}}$ is.

### Example: Web Design

![[Confidence/Screen_Shot 6.png]]

## Confidence Level Summary

- The phrase confidence level is used to describe the likeliness, chance or probability that a yet-to-be constructed interval will actually contain the true population value.
- Once we have “looked at” (computed) the actual interval, we can’t talk about probability or chance for this particular interval anymore. The population parameter is not a random quantity, it does not vary. **We can’t talk about the probability that the parameter is contained in an interval we already computed.**
- Therefore, the confidence level applies to the procedure, not to an individual interval; it applies “before you look” and not “after you look” at your data and compute your observed interval of values.

## Changing the Confidence Level

- You can change the confidence you have by changing how many SE you use in your range.
- For example, you could replace 1.98 with 2.58 to capture 99% of the observations. If you used 1.65, you would capture 90% of the observations.