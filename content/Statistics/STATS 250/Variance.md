Tags: January 27, 2021 9:40 AM

### Variance

Variance $\sigma^2$ is defined as: 

![[Variance/Screen_Shot.png]]

![[Expressing the variance using a compact form of the mean calculation]]

Expressing the variance using a compact form of the mean calculation

- Variance is the averaged squared distance between the observations and mean
- The variance is a measure of **variability**. Variability refer to how variable the data are compared to the mean.
- If you add a constant to all values, the variance remains the same.

### Standard deviation

The square root of the variance

![[Variance/Screen_Shot 2.png]]

- Incorrect: the standard deviation is the average distance between the observations. This is wrong because this would be a different equation:
    
    ![[Variance/Screen_Shot 3.png]]
    
- Note: you can change data in such a way that both the mean and variance change or only one changes.
- The standard deviation is a measure of variability ("wiggle room") from the mean.
- The **deviation** is how much an observation is different from the mean: deviation = observation – mean = 𝑥 − 𝑥̅
- *s* = 0 means there is no variability in the data. That is, all observations are the same (and are equal to the mean).

<aside>
💡 The standard deviation is expressed in the same units as the mean is, whereas the variance is expressed in squared units, but for looking at a distribution, you can use either just so long as you are clear about what you are using. The SD is usually more useful to describe the variability of the data while the variance is usually much more useful mathematically.

</aside>

### Sample variance and standard deviation

![[Variance/Screen_Shot 4.png]]

- If your data is a sample instead of a population, then you use:
    - For mean: $\bar{x}$ instead of $\mu$
    - For variance: $s^2$ instead of $\sigma^2$
    - For standard deviation: $s$ instead of $\sigma$.
- Additionally, you now divide by $n- 1$ instead of $n$.
    
    ![[Variance/Screen_Shot 5.png]]
    

```cpp
# This is R command for sample variance
# there is no command for population variance
var(c(x1, x2, ...))
```

### Misc:

- Observational units (aka cases) are the data collected. Ex. this could be the ratings for a professor and you get 10 cases if ten students reviewed the professor.

### Interquartile range

- The interquartile range is the distance between Q1 and Q3.
- IQR = Q3 - Q1
- IQR is robust to extreme observations

### Range

- The **distance between the maximum and minimum values** (100th percentile and 0th percentile).
- The range is very sensitive to outliers.

### Five number summary

- This is given by the minimum, Q1, median, Q3, and maximum

```r
x = c(1, 2, 3, ...)
# provides 5 number summary
summary(x) 
```