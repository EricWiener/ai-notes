### Difference between all the p's

- $p$ is the population proportion in the category of interest
- $\hat{p}$ (p-hat) is the sample proportion in the category of interest
- p-value is the probability of obtaining test results at least as extreme as the results actually observed, you could also think of this as the evidence against the null hypothesis with a smaller p-value leading to stronger evidence against the null
- $p_0$ (p-null) is the hypothesized value of the population proportion that we use to build the null model, p-null comes from the research question and not the data

# Question 1

An intelligence quotient (IQ) is a total score derived from a set of standardized tests or subtests designed to assess human intelligence. For a standard modern IQ test, the raw score is transformed to a normal distribution with mean 100 and standard deviation 16.

**a. It is frequently stated that approximately two thirds of the population have IQs between 85 and 115. Assess the validity of the statement.**

We can use the 68–95–99.7 to estimate this. If the population's IQ distribution follows a normal model, then 68% of the population's IQs will be within 1 standard deviation of the mean. The mean is 100 and the standard deviation is 16. Therefore, approximately 68% of the population should have an IQ between 84 and 116, so it would make sense that approximately 66% (two thirds) have an IQ between 85 and 115.

Compute the area between 85 and 115 under N(100, 16) curve:
`pnorm(115,mean=100,sd=16) - pnorm(85,mean=100,sd=16)=0.6514986`
This is not far from 2/3.

**b. The elite society Mensa only admits individuals who score in the top 2% of the population. What is the minimum IQ score of the members?**

`qnorm(0.98, mean = 100, sd = 16, lower.tail = TRUE)`

132.86

*IQ scores usually don’t contain fractions. So we could round it to 133.*

**c. The score distribution of a more specialized IQ test is normal with mean 150 and standard deviation 20. It is believed that the specialized test is comparable to the common test in the sense that a test taker’s scores will be placed at roughly the same percentile based on the two tests. If someone scores 110 points on the common test, how many points are they expected to score on the specialized test?**

$z = \frac{x - \mu}{\sigma}$

- If they scores 110 points on common test, this is:
    - $z = \frac{x - \mu}{\sigma} = \frac{110 - 100}{16} = \frac{10}{16}$
- Therefore, we can get $x$ on the standardized test if we do:
    - $\frac{10}{16} = \frac{x - 150}{20}$, which gives us $x = 162.5$

# Question 2

Overall, more Americans prefer watching the news to reading the news. But does that preference vary by age? To explore this question, we took large, independent random samples of young American adults (18-34 years old) and of older American adults (50+ years old) and asked each if they prefer watching the news to reading the news. Using the sample results, we constructed a 99% confidence interval to estimate the difference $p_Y - p_O$, where $p_Y$ = the population proportion of young American adults who prefer watching the news to reading the news, and $p_O$ = the population proportion of older American adults who prefer watching the news to reading the news. The resulting 99% confidence interval is (−0.005, 0.258).

Determine if each of the following statements is correct or incorrect.

**Parameter of interest is** $p_Y - p_O$

1. **If we repeated this procedure many times and for each repetition we computed the 99% confidence interval, we would expect 99% of the resulting intervals to contain the difference in the sample proportion of young American adults who prefer watching the news to reading the news less the sample proportion of older American adults who prefer watching the news to reading the news.**

Incorrect because we would expect 99% of the intervals to contain the difference in the **population proportion** (not sample proportion).

2. **One assumption required for constructing the confidence interval for the difference in population proportions is that each population of responses, preference in getting the news, follows a normal distribution.**

Incorrect. The responses are categorical (yes or no) so the distribution can't be normal. Only continuous data can be normal.

3. **Because most of the reasonable values in the 99% confidence interval are positive, we can say that population proportion of young American adults who prefer watching the news to
reading the news is significantly higher than the population proportion of older American adults who prefer watching the news to reading the news.**

False. We do not know where in the confidence interval the true difference in proportions lies (or if it is necessarily in the confidence interval). Because the confidence interval includes both negative and positive values, we can't say that the proportion is different. 

Additionally, the qualification "significantly higher" needs to be defined. Ex. if you want the proportion to be different by 25%, you would need to test that hypothesis.

- $H_0: p_Y - p_O \leq 25%$
- $H_A: p_Y 0- p_O > 25%$

# Question 3

To study the adoption of streaming service by American households, a recent CNBC All-American Economic Survey polled 801 Americans around the country and found that 57% of them had some form of streaming service, compared with 43% who do not. They then proceeded to construct a confidence interval for the true proportion p of American households that have some form of streaming service.

**a. What is the point estimate of p based on our data?**

A point estimate is a number that is used to estimate the parameter.

$\hat{p} = 0.57$

**b. What is the standard error of the sample proportion** $\hat{p}$**? What is an estimate of this standard error?**

$SE_{\hat{p}} = \sqrt{\frac{p(1-p)}{n}}$. We don't know the true proportion $p$, so we can estimate it by $SE_{\hat{p}} = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$, which is $SE_{\hat{p}} = \sqrt{\frac{0.57(0.43)}{801}}$

![[Exam 2 Rev/Screen_Shot.png]]

**c. What are the conditions required for computing the confidence interval? Check those conditions.**

1. Random sampling has occurred
2. We have a sufficiently large sample size, meaning ALL of the following are true:
    1. $n\hat{p}\geq 10$
    2. $n(1 - \hat{p}) \geq 10$

<aside>
💡 You check sample size with $\hat{p}$ for confidence intervals. For hypothesis test, you use $p_0$, the null hypothesis value (which is provided by the research question).

</aside>

![[Exam 2 Rev/Screen_Shot 1.png]]

**d. Use the result to construct a 95% confidence interval for p. What’s the margin of error (ME) of the interval?**

`prop_test(x = 0.57 * 801, n = 801, conf.level = 0.95)`

![[Exam 2 Rev/Screen_Shot 2.png]]

$$
\text{point estimate} \pm \text{margin of error}
$$

- Centered at the point estimate
- Margin of error is 1/2 the width of the confidence interval (CI).
- Therefore, margin of error is (0.6042849 - 0.5357151) / 2 = 0.0342849

**e. In order for the margin of error to be at most 0.03, how large a sample should we have?**

The confidence interval is given by $\hat{p} \pm z^*\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$

We have a margin of error of 0.0342849. The confidence interval is 95%, so we use $z^* = 1.96$, we know $\hat{p}$, so we just need to solve for $n$. You should round $n$ up to make sure the margin of error is at most 0.03.

- If the confidence level = 90%, $z^* = 1.65$
- If the confidence level = 95%, $z^* = 1.96$
- If the confidence level = 99%, $z^* = 2.58$

# Question 4

4. According to the Center for Disease Control (CDC), the percent of adults 20 years of age and over in the United States who are overweight is 69.0%. One city’s council wants to know if the proportion of overweight citizens in their city is different from this known national proportion. They take a random sample of 150 adults 20 years of age or older in their city and find that 98 are classified as overweight.

**a. State the null and alternative hypotheses.**

$H_0: p = 0.69$

$H_A: p \ne 0.69$

Let $p$ = the percent of adults 20 years of age and over in the city who are overweight.

![[Exam 2 Rev/Screen_Shot 3.png]]

**b. Check the conditions for performing the test.**

1. Random sampling has occurred
2. We have a sufficiently large sample size, meaning ALL of the following are true:
    1. $np_0\geq 10$
    2. $n(1 - p_0) \geq 10$

![[Exam 2 Rev/Screen_Shot 4.png]]

**c. Calculate the test statistic and determine the p-value.**

$\hat{p} = \frac{98}{150}$.

```cpp
p_hat <- 98/150
prop_test(x = 98, n = 150, p = 0.69, conf.level = 0.95)
```

![[Exam 2 Rev/Screen_Shot 5.png]]

Z = - 0.9709831, p-value is 0.3316

![[You could also manually calculate the test statistic]]

You could also manually calculate the test statistic

**d. Evaluate the p-value and the compatibility of the null model.**

The p-value is > .010. So there is little evidence against $H_0$.

**e. Make a conclusion in the context of the problem.**

Since our p-value is (**large**), we have (**weak**) evidence against the null hypothesis. Thus we have (**weak**) evidence to support that the true proportion of overweight citizens in their city is different from the known national proportion of 0.69.

![[Exam 2 Rev/Screen_Shot 7.png]]

# Question 5

5. A medical researcher is interested in understanding if smoking can result in the wrinkled skin around the eyes. The researcher recruited 150 smokers and 250 nonsmokers to take part in an observational study and found that 95 of the smokers and 105 of the nonsmokers were seen to have prominent wrinkles around the eyes (based on a standardized wrinkle score administered by a person who did not know if the subject smoked or not). Suppose the researcher decides to construct a confidence interval to estimate the different between the true proportions of those in the populations of smokers and nonsmokers who have wrinkled skin around the eyes. Let $p_S$ and $p_N$ be the true proportions and $\hat{p}_S$ and $\hat{p}_N$ be the sample proportions.

**a. What is our point estimate of $p_S - p_N$ ?**

The point estimate of $p_S - p_N$ is $\hat{p}_S - \hat{p}_N$ = 95/150 − 105/250 = 0.2133.

**b. What is the standard error of $\hat{p}_S - \hat{p}_N$ and what is an estimate of the standard error?**

![[Exam 2 Rev/Screen_Shot 8.png]]

```cpp
sample_size_s <- 150
sample_size_n <- 250
sqrt(((p_hat_s * (1-p_hat_s)) / sample_size_s) + ((p_hat_n * (1-p_hat_n)) / sample_size_n))
```

**c. What conditions do we need to examine before proceeding to construct the confidence interval?**

1. **Independence within each sample**: The observations within each sample are independent (e.g., we have a random sample from each of the two populations).
2. **Independence between the samples**: The two samples are independent of one another such that observations in one sample tell us nothing about the observations in the other sample (and vice versa).
3. **Success-failure condition**: Both samples must satisfy the success-failure condition. That is $n_1 \hat{p}_1, n_1(1 - \hat{p}_1), n_2\hat{p}_2, n_2(1 - \hat{p}_2)$ are all at least 10.

**d. Use the data to construct a 99% confidence interval of $p_S - p_N$ .**

(0.08396237, 0.34270430)

```cpp
prop_test(x = c(95, 105), n = c(150, 250), conf.level = 0.99)
```

![[Exam 2 Rev/Screen_Shot 9.png]]

![[Exam 2 Rev/Screen_Shot 10.png]]

# Question 6

A researcher suspects that the rate of failing the AP-Statistics exam is higher for rural students than for suburban students. She randomly selects 107 rural students and 143 suburban students who took the exam. Thirty rural students failed to pass their exam, while 45 suburban students failed the exam.

**a. State the null and alternative hypotheses to address the research question. Define the relevant symbols and state the hypotheses both in words and in symbols.**

Let $p_r, p_s$ be the true proportions of rural and suburban students who fail the exam. The hypotheses are

$H_0: p_r - p_s = 0$, namely, rural students and suburban students fail the AP-Statistics exam at the same rate

$H_A : p_r - p_s > 0$, namely, rural students fail the AP-Statistics exam at a higher rate than do suburban students

**b. Check the conditions for performing the test.**

![[Exam 2 Rev/Screen_Shot 11.png]]

![[Exam 2 Rev/Screen_Shot 12.png]]

**c. Calculate the test statistic and determine the p-value.**

```cpp
prop_test(x = c(30, 45), n = c(107, 143), conf.level = 0.95, alternative = "greater")
```

![[Exam 2 Rev/Screen_Shot 13.png]]

- Note that when you choose the alternative, it is one of "greater", "less", or "beyond". This refers to $p_r - p_s > 0$ which is saying the difference is greater than 0. You order your inputs to `x` and `n` in the order $r, s$.

![[Exam 2 Rev/Screen_Shot 14.png]]

**d. Evaluate the p-value and the compatibility of the null model.**

The p-value is > 0.1, so it does not suggest that the null model is violated.

**e. Make a conclusion in the context of the problem.**

There is no evidence against the null hypothesis. The data do not support the claim that rural students fail the AP-Statistics exam at a higher rate than do suburban students.