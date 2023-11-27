### Question 5 seems to have two options that are correct (A and B). B seems like a better answer, but A also looks like its correct. Is there a reason A isn’t correct?

![[Exam 2 Not/Untitled.png]]

Answer choice A is not a conclusion, it's a decision. A conclusion is a thing you tell your non-statistician friends about; in this case, that the mean number of parking tickets issued in Ann Arbor is less than 400/day. Your friends who haven't taken a stats class don't know what it means to "reject the null hypothesis", but they do understand what an average number of parking tickets is. Also recall that we're not using terms like "reject" or "fail to reject" in STATS 250 -- we find them to be overly reductive.

## Choosing $p$, $p_1 - p_2$, or $\mu$

- Do college freshmen study more than 10 hours per week on average?
    - $\mu$ because it asks about on average
- Percentage of all such consumers who prefer the grape flavor over the apple flavor.
    - $p$ because it asks about a single percentage
- Is there a difference in the percentage of satisfaction of receiving a gift when receiving clothes as a gift versus receiving electronics as a gift?
    - $p_1 - p_2$ because it's the difference of two proportions
- Are the majority of college freshmen studying from their parents house?
    - $p$ because this is a proportion

## Confidence Interval vs. Confidence Level

- **Problem context**
    
    Pokémon Go is a reality game launched by Nintendo in 2016. Industry sources believe that
    70% of all U.S. Pokémon Go players are young (less than 34 years of age). A researcher plans
    to survey a random sample of Pokémon Go players in Ann Arbor and estimate the population
    proportion of all Ann Arbor Pokémon Go players who are young. The researcher takes a large
    random sample of Ann Arbor Pokémon Go players. The 98% confidence interval is 60% to 68%.
    
1. Provide an interpretation of the 98% **confidence interval** in context.
    
    We estimate with 98% confidence that the true proportion of Ann Arbor Pokemon Go
    players who are young is between 60% and 68%.
    
2. Provide an interpretation of the 98% **confidence level** in context.
    
    If we repeated this procedure many times, we would expect that 98% of the created
    confidence intervals would contain the true proportion of Ann Arbor Pokemon Go players
    who are young.
    

<aside>
💡 The 90% INTERVAL means that we are 90% confident that the true population parameter (whether that’s p, p1-p2, or mu) is within the confidence interval. The confidence LEVEL, however, means that if we redid the sample a large number of times and construct a confidence interval each time, we expect 90% of the confidence intervals to contain the population parameter.

</aside>

## Calculating Quantile

- Question Context
    
    Alysha scored 670 points on the Mathematics part of the SAT. SAT Math scores for that year were approximately normally distributed with mean of 516 points and standard deviation of 116 points. John took the ACT and scored 26 on the Mathematics portion. ACT Math scores for that year were approximately normally distributed with mean of 21.0 points and standard deviation of 5.3 points.
    

**Alysha’s score corresponds to the ____ quantile for the SAT Math distribution.**

```r
# Calculating the quantile (area under the curve) using score
pnorm(670, mean = 516, sd = 116)

# Calculating the quantile (area under the curve) using z-score
z_score <- (670 - 516)/116
pnorm(z_score)
-------------------
[1] 0.9078426
[1] 0.9078426
```

- The quantile is the area under the curve

**Assuming both tests measure the same type of ability, who had a better test result? Include numerical support and your work.**

```r
pnorm(q = 670, mean = 516, sd = 116)
[1] 0.9078426
pnorm(q = 26, mean = 21, sd = 5.3)
[1] 0.8272609
```

Thus Alysha did better, because she scored in the 90.7th quantile, while John only
scored in the 82.7th quantile.

**All students who took the SAT Math test and scored in the top 2% received a letter from the National Honor Society inviting them to join. What is the lowest SAT Math score needed to be invited?**

```r
qnorm(p = 0.98, mean = 516, sd = 116)
[1] 754.2349
```

**Complete the following statement based on what is known about the distributions: Approximately 68% of all ACT Math scores are between _ points and _ points.**

```r
qnorm(p = ((1.0 + 0.68)/2), mean = 21, sd = 5.3)
qnorm(p = ((1 - 0.68)/2), mean = 21, sd = 5.3)
-------------
[1] 26.27063
[1] 15.72937
```

You could also just use the fact that 68% of the data are within 1 standard deviation and do 21 - 5.3 = 15.7 and 21 + 5.3 = 26.3.

## Experiment vs. Observation

**Why it's an observational study**

An experiment would require that we randomize participants into two groups, and control for any confounding variables. Since this is just a simple random sample of collected data, it is an observational study.

**Why it's an experiment**

Participants are randomized into two groups, a treatment, and a control group. The treatment group will keep a journal, and the control group will do nothing. This is an example of an experiment.

## Calculate confidence interval for difference in proportions

- Question Context
    
    One proposal being voted on in an election is Proposal C: Should a tax be authorized for street and bridge resurfacing and reconstruction? Mr. Young is conducting a survey among residents of the city and a survey among residents of the county (who live outside the city limits) to estimate the true difference in the proportions of city residents who will favor Proposal C and county residents (who live outside the city limits) who will favor proposal C. Mr. Young has randomly selected 245 city residents and 132 county residents, and, of these, 198 and 91 are in favor of Proposal C, respectively. Mr. Young wishes to compute a confidence interval for the difference in proportions of those who favor Proposal C between the two groups, city residents and county residents.
    

**Use the results of the survey to compute a 99% confidence interval for the true difference in proportions of those who favor Proposal C between the two groups, city residents and county residents**

```r
prop_test(x=c(198, 91), n=c(245, 132), conf.level=0.99)

----------------
2-sample test for equality of proportions without continuity correction

data:  x out of n
Z = 2.6005, p-value = 0.009309
alternative hypothesis: two.sided
99 percent confidence interval:
 -0.003548507  0.241087159
sample estimates:
   prop 1    prop 2 
0.8081633 0.6893939
```

The 99% confidence interval is (-0.004, 0.241).

**Provide an interpretation of the 99% confidence interval, in context.**

We are 99% confident that the true difference in proportions of those who favor Proposal C between the two groups, city residents and county residents, is between -0.004% and 24.1%.

## Finding p-value given test statistic

You can just plug it into `pnorm` with N(0, 1). Example: if you want to find the p-value for the corresponding test statistic when you do a hypothesis test for the difference between two proportions:

```r
pnorm(q = 1.73, mean = 0, sd = 1, lower.tail = FALSE) * 2
```

- Notice we do `lower.tail = FALSE` in order to find the area beyond the test-statistic.
- We then double this value because the tail is two-sided.

## Probability of the null hypothesis

The null hypothesis is either true (100% probability) or false (0% probability). And we do not know which one. When doing hypothesis tests, we calculate a p-value, which has a different definition.

The $p$-value is defined as the probability of obtaining a value of the statistic at least as extreme as the observed statistic when the null hypothesis is true.

## When can you have type 1 or 2 error

- Type 1 Error: the null hypothesis ($H_0$) is true, but we falsely reject it. **We reject** $H_0$.
- Type 2 Error: the null hypothesis ($H_0$) is false, but we fail to reject it. **We fail to reject** $H_0$.

Example: your p-value is 0.08. Can you have a type 2 error?

- With a p-value of 0.0836, we would have some evidence against $H_0$. As such, only a Type 1 error is possible. To make a Type 2 error, we would need to have a p-value that is larger that 0.1.
- This is because a type 2 error requires not rejecting the null hypothesis
- since our p value is 0.0836, we have some evidence against H0. This would mean that we would reject the null hypothesis because "some" in this case would mean to reject the null. So, with that in mind, we would never commit a Type 2 error because we would never FAIL to reject it, since we already ARE ALWAYS rejecting the null based on the p value. If the p value is greater than 0.1, then that would mean that we would have little evidence to reject the null, which means that we WOULD FAIL to reject it, since the p value is too large to be able to consider the alternative hypothesis as true.
- A type II error can occur only when you have NO evidence against the null hypothesis. Here, we have some evidence, so we can’t make a type-II error.

## Does a confidence interval make sense?

For a confidence interval to make sense, it needs to be centered at the point estimate. For a difference of proportions, the point estimate is $\hat{p}_1 - \hat{p}_2$. For a single proportion, it is centered at $\hat{p}$.

**Testing confidence interval difference of proportions:**

```r
prop_test(x=c(0.42 * 600, 0.36 * 300), n=c(600, 300), conf.level = 0.9)
```

![[Exam 2 Not/Screen_Shot.png]]

**Testing confidence interval single proportion**

```r
prop_test(x = # successes, n = sample size, conf.level = confidence level)
```

## Single proportion hypothesis test example

In 2016, it was stated that 64% of the University of Michigan's incoming freshmen 
were from the state of Michigan (called ‘in-state’). For a research project, an undergraduate student, Jessie, proposes to assess if the current proportion of all students in the 2020 incoming freshmen class has changed from this previous level. In a random sample of 150 current U of M freshmen, 86 of the sampled freshmen are ‘in-state’. Complete the following questions, showing your work, and upload a .jpg or PDF file with all 5 parts.

**a. State the null and alternative hypotheses both in symbols and in words.**

$H_0$: The proportion of UM freshmen in 2020 who are in-state is equal to 64% (p = 0.64) 

$H_a$: The proportion of UM freshmen in 2020 who are in-state is not equal to 64% (p != 0.64)

**b. Check any conditions necessary for the hypothesis test.**

Two conditions:

1. Independence within the group: since we took a random sample, this condition is
reasonably met.
2. We have a sufficiently large sample size, meaning ALL of the following are true:
    1. $np_0\geq 10$
    2. $n(1 - p_0) \geq 10$
    
    Here $p_0 = 0.64$ and $n = 150$
    

**c. Calculate the test statistic, showing any relevant work.**

```r
prop_test(x = 86, n = 150, p=0.64)
```

![[Exam 2 Not/Screen_Shot 1.png]]

Test statistic is Z = -1.701

**d. Calculate the p-value and comment on the amount of evidence against the null hypothesis.**

p-value is 0.8894. We have some evidence against the null hypothesis.

**e. State your conclusion in context.**

We have some evidence to suggest that the proportion of UM in-state students in 2020 is different from 36%, the value from the year 2016.