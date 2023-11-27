Files: lab12-slides-async.pdf

<aside>
💡 Multiple regression extends simple two-variable regression to the case that still has one response but many explanatory variables ($x_1, x_2, x_3, ...$). The method is motivated by scenarios where many variables may be simultaneously connected to an outcome.

</aside>

## Fitting a multiple linear regression model

```cpp
mod1 <- lm(body_mass_g ~ flipper_length_mm + species, data = penguins)
summary(mod1)
```

![[Lab 12 Mul/Screen_Shot.png]]

- Formula notation is `response variable ~ explanatory variable + explanatory variable`.
- You can add explanatory variables to our model `+`

### **Coefficients for a multiple linear regression model**

![[Lab 12 Mul/Screen_Shot 1.png]]

- The variables `speciesChinstrap` and `speciesGentoo` are the indicator variables for the species.
- When the penguin species is Adelie, `speciesChinstrap` = 0 and `speciesGentoo` = 0.
- When the penguin species is Chinstrap, `speciesChinstrap` = 1 and `speciesGentoo` = 0.
- When the penguin species is Gentoo, `speciesChinstrap` = 0 and `speciesGentoo` = 0.

The equation for the regression line is: $\hat{y} = b_0 + b_1 x_1 + b_2 x_2 + b_3 x_3$

- where $\hat{y}$ represents **the predicted body mass (g)**, x1 represents **the flipper length (mm)**, x2 represents **if the species was Chinstrap**, and x3 represents **if the species was Gentoo**.

### **Reference Level**

<aside>
💡 Note that you only need two additional rows of coefficients even though the variable has three levels. When both indicators are zero, this is when you have the other level.

</aside>

- In the above example, the Adelie penguins were the reference level
- For Adelie penguins, **both** speciesChinstrap and speciesGentoo are zero. Adelie
penguins are the **reference level**.
- R chooses the reference level *alphabetically*.

## Regression Conditions

There are **four** conditions under which the simple linear regression line is the line of best fit:

- **Linearity:** The relationship between the explanatory and response variables should be linear.
- **Independence:** The observations must be independent of one another. This does not
mean that the response and explanatory variables are independent; rather, that the
"individuals" from whom we collect information must be independent of each other.
- **Nearly Normal Residuals:** The residuals should come from a nearly-normal
population of residuals.
- **Equal (constant) variability:** The variability of the residuals should not depend on where they are along the regression line.

### Residuals vs. Fitted Values Plots

```cpp
plot(mod1$residuals ~ mod1$fitted.values,
     main = "Residuals vs. Fitted Plot",
     xlab = "Fitted Values (body mass on flipper length and species",
     ylab = "Residuals")
```

![[Lab 12 Mul/Screen_Shot 2.png]]

- **Linearity**: demonstrated by symmetry around the horizontal line $y = 0$
    - There is some noticeable clustering. This is expected with categorical predictors. Linearity looks good here.
- **Equal variance**: demonstrated by similar spread of points across the plot
    - Relatively even spread along that curved path of most of the points. Looks good.
    - You want a constant band of points. You don't want points that widen or narrow as you move along the x-axis.

```cpp
qqnorm(mod1$residuals,
       main = "QQ Plot of Residuals for Body Mass on Flipper Length \nand Species")
qqline(mod1$residuals)
```

![[Lab 12 Mul/Screen_Shot 3.png]]

- Nearly normal: we're looking for a straight line with a positive slope
    - Some deviations are expected and perfectly straight lines are suspicious
    - In this QQ plot, most points are on that straight line and deviation is expected in the tails
    - We can reasonably conclude that the **population of residuals** is normally distributed