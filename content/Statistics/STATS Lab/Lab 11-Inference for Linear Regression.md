Files: lab11-slides-async.pdf

### Paired $t$ tests

- We calculate these with the same equations as a $t$ test for one population mean
- We'll test the hypothesis:
    
    $$
    H_0: \mu_{\text{diff}} = 0 \text{ vs. } H_a: \mu_{\text{diff}} \ne 0
    $$
    
- Ex: $\mu_{\text{diff}}$ is the population mean of the difference in father's age and mother's age (father minus mother).
- To run this in R, we need to create a variable that represents these differences.
- We end up with one variable (the mean differences) and we just do a $t$ test for one population.

### Calculate difference variable

```r
births <- read.csv("births.csv", stringsAsFactors = TRUE)

# Calculate the difference between female age
# and male age. Store in new variable diff
births$diff <- births$fage - births$mage
```

### Run a `t.test()` on the differences variable.

```r
t.test(births$diff, mu = 0, alternative = "two.sided")
```

![[Lab 11 Inf/Screen_Shot.png]]

- This is called one sample t-test in R because to R that's all we did.
- We have strong evidence to suggest that the mean of the population differences is not zero.

## Alternative paired t-test

- Can skip having to manually calculate the difference variable
- Can use the `paired` argument in `t.test()` to allow passing two arguments with data.
- R will calculate the difference for us with `x - y`
- We get the same $t$, $df$, and p-value

```r
t.test(x = births$fage, y = births$mage, paired = TRUE,
       mu = 0, alternative = "two.sided") 
```

![[Lab 11 Inf/Screen_Shot 1.png]]

<aside>
💡 R incorrectly states the alternative hypothesis. It should day "true mean of differences is not equal to 0".

</aside>

# Linear regression

- Exploring linear regression with penguin data.
- You should first make a scatterplot to look at the data.

### Plotting data

```r
penguins <- read.csv("penguins.csv", stringsAsFactors = TRUE)
plot(body_mass_g ~ flipper_length_mm,
     data = penguins,
     ylab = "Body Mass (g)", xlab = "Flipper Length (mm)",
     main = "Penguin Body Mass vs. Flipper Length")
```

![[Lab 11 Inf/Screen_Shot 2.png]]

- Note that we plot `body_mass_g ~ flipper_length_mm`. You always do `y ~ x`

### Fitting a linear regression model

```r
# We call this with the arguments y ~ x
mod1 <- lm(body_mass_g ~ flipper_length_mm, data = penguins)
summary(mod1) # need summary() to get output
```

![[Lab 11 Inf/Untitled.png]]

- The top box has $b_0$ (the intercept) and $b_1$ (the slope)
- The bottom box has the $R^2$ value.
- The slope of 50.15 allows us to say: "We estimate that a one-millimeter longer flipper is associated with **50.15**-gram **higher** body mass, on average, in the population of penguins represented by this sample."

# Regression Diagnostics

There are four conditions under which the simple linear regression line is the line of best fit:

- **Linearity:** The relationship between the explanatory and response variables should be linear.
- **Independence:** The observations must be independent of one another. This does not mean that the response and explanatory variables are independent; rather, that the "individuals" from whom we collect information must be independent of each other.
- **Nearly Normal Residuals:** The residuals should come from a nearly-normal population of residuals.
- **Equal (constant) variability:** The variability of the residuals should not depend on where they are along the regression line.

### Checking conditions using diagnostic plots

- We're going to use the fact that `mod1` has a lot of info in it (it is a dataframe).
    
    ![[Lab 11 Inf/Screen_Shot 3.png]]
    
- We're going to use `residuals` and `fitted.values`

### Residuals plot (aka residuals vs. fitted values)

```r
# y vs. x (residuals will be on the y-axis)
plot(mod1$residuals ~ mod1$fitted.values,
     main = "Residuals vs. Fitted Values",
     ylab = "Residuals",
     xlab = "Fitted Values (body mass vs. flipper length)")
abline(h = 0) # draw a line at 0
```

![[Lab 11 Inf/Screen_Shot 4.png]]

**What conditions does this plot help us check?**

- **Linearity:** demonstrated by symmetry around the horizontal line $y = 0$
    - This plot has some slight curvature. Not awful, but something to look into.
- **Equal variance:** demonstrated by similar spread of points across the plot.
    - This plot looks good.

### QQ Plots

```r
qqnorm(mod1$residuals)
qqline(mod1$residuals)
```

![[Lab 11 Inf/Screen_Shot 5.png]]

- Remember nearly normal conditions applies to residuals - not the values themselves.

**What does this plot help us assess?**

- The nearly normal condition. We are looking for a straight line with a positive slope. Some deviations are expected.
    - This plot looks good. Most points are on the straight line and deviation is expected in the tails.
- We can reasonably conclude that the population of residuals is normally-distributed.

# Inference for linear regression

Let's say we want to know if, at the population level, there's a linear relationship between penguin flipper length and body mass. If there's no relationship, then $\beta_1$, the population slope, should be 0. So our hypotheses are:

$$
H_0: \beta_1 = 0 \quad \text{vs.} \quad H_a: \beta_1 \neq 0.
$$

![[Lab 11 Inf/Screen_Shot 6.png]]

- Can use the output of `summary(mod1)` to find the values you need to calculate the test statistic.
- $SE_{b_1}$is the standard error for the `x` variable.
- The null value is 0.
- The $t$ value we calculate is the same as the value given in the `t value` column for the `x` variable. That `t value` is the test statistic of whether the parameter for that row is equal to 0.
    - Ex: 32.56 is the test statistic for the t-test of whether your slope is 0
    - -18.924843 is the test-statistic for the t-test of whether the intercept is 0.

### Plotting results

```r
plot_t(df = 333 - 2, shadeValues = c(-32.56, 32.56), direction = "beyond")
```

![[Lab 11 Inf/Screen_Shot 7.png]]

- We use `n - 2` degrees of freedom
- This is a two-sided test, so you shade beyond the test statistics.

### Calculating p-value

```r
2 * pt(q = -32.56, df = 333 - 2)
[1] 3.186042e-105
```

- It's a two-sided test, so we need to multiply the area we get by two.
- Taking the area to the left of the smaller test statistic and multiplying it by two.

<aside>
💡 Note that the answer we got matches up with the output of the summary table (shown below). R will give you the **two-sided** p-values for the hypothesis test that the parameter is zero. If you want a one-sided p-value, you need to divide by two.

</aside>

![[Lab 11 Inf/Screen_Shot 6.png]]

## Confidence intervals for regression parameters

Confidence interval for a population regression slope:

$$
b_1 \pm t^* \times \text{SE}_{b_1}
$$

- You can also calculate a confidence interval for the intercept. You just need to replace all the $b_1$ with $b_0$

```r
# pass in linear regression model and the confidence level
confint(mod1, level = 0.95)
```

![[Lab 11 Inf/Screen_Shot 8.png]]

- The `2.5%` column is the lower limit for the confidence interval (2.5% quantile)
- The `97.5%` column is the upper limit for the confidence interval. (97.5% quantile)

## Plotting regression line

```r
plot(body_mass_g ~ flipper_length_mm,
     data = penguins,
     ylab = "Body Mass (g)", xlab = "Flipper Length (mm)",
     main = "Penguin Body Mass vs. Flipper Length")

# Need to pass in the linear regression model
# as the argument to abline()
abline(mod1, col = "tomato", lwd = 2)
```