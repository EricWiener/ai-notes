Created: July 15, 2020 7:49 AM
Type: Notes

When combining two Gaussian Distributions (the two in red), you will get a peakier distribution that has a peak between the two other peaks. In the picture below, the two red Gaussians combine into the blue Gaussian. The more measurement you get, the less uncertain you are.

![[Multiplyin/Screen_Shot.png]]

If the short, red Gaussian is parameterized by $\mu_1$ and $\sigma^2_1$ and the taller, red Gaussian is parameterized by $\mu_2$ and $\sigma^2_2$, the product of the two red Gaussians is parameterized with:

- Mean: $\frac{\mu_1\sigma^2_2 + \mu_2\sigma^2_1}{\sigma^2_2 + \sigma^2_1}$ - note that the means in the numerator are multiplied by the other distribution's variance.
- Variance: $\frac{\sigma^2_1\sigma^2_2}{\sigma^2_1 + \sigma^2_2} = \frac{1}{\frac{1}{\sigma^2_1} + \frac{1}{\sigma^2_2}}$

Note that if you add two Gaussians with the same variance, the new variance will be $\frac{\sigma^2}{2}$, so the new distribution is narrower.