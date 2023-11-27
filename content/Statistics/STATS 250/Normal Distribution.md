Files: normal-1.pdf
Tags: February 22, 2021 7:55 AM

# Working with normals in R

### Calculating area of the under the normal curve

You can use R to do this for any value $x$ on the x-axis:

```r
# will give area to left of x under curve N(mu, sigma)
pnorm(x, mean = mu, sd = sigma)
```

![[Normal Dis/Screen_Shot.png]]

Conversley, for an probability $p$, a probability between [0, 1], you can find the point $q$ on the x-axis such that the area to the left of $q$ is $p$ with the command `qnorm`

- $q$ is the (100 * p)th percentile, which is also referred to as the $p$-th quantile of the distribution

```r
qnorm(p, mean = mu, sd = sigma)
```

![[Here, qnorm(0.5, mean = 0, sd = 1) would return 0, which is the point that has 50% of the distribution to the left of it.]]

Here, qnorm(0.5, mean = 0, sd = 1) would return 0, which is the point that has 50% of the distribution to the left of it.

## Example: Golden Retrievers

![[Normal Dis/Screen_Shot 2.png]]

![[Normal Dis/Screen_Shot 3.png]]

![[You can set lowertail = FALSE to get the upper area]]

You can set lowertail = FALSE to get the upper area

![[Normal Dis/Screen_Shot 5.png]]

# 68-95-99.7 rule

![[Normal Dis/Screen_Shot 6.png]]

- Values that fall more than 3 standard deviations from the mean are very rare.

![[Normal Dis/Screen_Shot 7.png]]