Files: lab09-slides-async.pdf

- We've spent the past month learning how to answer questions about population **proportions**.
    - Proportions arise from categorical data
    - Ex: the proportion of college students who are 20 years old (drinking alcohol is a binary yes or no).
- Now we're going to shift our focus to population **means**
    - Means arise from continuous data
    - Population mean denoted $\mu$
- The parameters, point estimates, and distributions change
    - New parameter: $\mu$
    - New point estimate: $\bar{x}$
    - New distribution $t(\text{df})$
    

## The t distribution

- Bell shaped
- Heavier tails than the normal distribution
- Used to approximate N(0, 1)
- "Indexed" by **degrees of freedom** df.
    - df is closely related to n
    - As df increases, t(df) gets closer to N(0, 1)
    
    $$
    t = \frac{\bar{x}-\mu}{s/\sqrt{n}}
    $$
    
    - For the normal distribution, we identified the distribution with $\mu$ and $\sigma$. Now, we just identify the distribution with $t$.
    - $t$ is called the t-statistic
    - It is essentially a standard score
    - You calculate it by dividing by $s/\sqrt{n}$ which is the standard error of $\bar{x}$.
        - Note we use $s$ here which is the sample standard deviation (not the population standard deviation).
    

### Why do we calculate the t-statistic using the standard error $s / \sqrt{n}$

- When working with a proportion $p$, if we know $p$, we know $\text{SE}_{\hat{p}}$.
    
    $$
    \text{SE}_{\hat{p}} = \sqrt{\frac{p(1-p)}{n}}
    $$
    
    - Here we can use $p$ and $n$ to calculate $\text{SE}_{\hat{p}}$
- With continuous data, knowing $\mu$ tells us nothing about $\sigma$.
    - So we need to use data to estimate $\sigma$ with $s$
    - The $t$ distribution is used here because we're estimating $\sigma$

# The $t$ Distribution

## `pt()`

We can find probabilities related to the $t$ distribution using the `pt()` function.

`pt()` works just like `pnorm()` except you pass `df` for degrees of freedom.

```cpp
pt(q = 1.4, df = 4)
[1] 0.8829497
```

- This tells you area to the left of `q` under the t distribution

## `plot_t()`

![[Lab 09 Inf/Screen_Shot.png]]

- Could also use `"greater"` and `"beyond"` as arguments for `direction`

## `qt()`

- You can use `qt()` to find a quantile of the t distribution (i.e. undo `pt()`)

![[Lab 09 Inf/Screen_Shot 1.png]]

- The value on the x-axis such that 0.8829497 of it is to the left of it (with a t distribution with 4 degrees of freedom) is 1.4

# Confidence Intervals

![[Lab 09 Inf/Screen_Shot 2.png]]

- Range of reasonable values for the parameter
- When construction the confidence interval, we have (point estimate) $\pm$ (a few aka $t^*$) ($SE_{\text{point estimate}}$)

### Conditions for inference on a population mean:

- Independence: we need a random sample of observations
- Nearly normal: our observations need to come from a population with a nearly normal distribution
    - Note even if the nearly normal condition is not satisfied, we can use the central limit theorem to avoid this being a showstopper: the **central limit theorem** tells us that for large enough samples ($n \geq 30$), $\bar{x}$ will be approximately $N(\mu, s/\sqrt{n})$. We can use this if the data doesn't appear to be normal.

![[Lab 09 Inf/Screen_Shot 3.png]]

- You can use `t.test()` instead of `prop_test()` to get a confidence interval.
- You can ignore the top output since we aren't using a hypothesis test

# Hypothesis test for one mean

![[Lab 09 Inf/Screen_Shot 4.png]]

- Here we use $\mu_0$ which is given by our null hypothesis

![[Lab 09 Inf/Screen_Shot 5.png]]