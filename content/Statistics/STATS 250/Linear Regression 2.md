Files: Regression_II.pdf
Tags: February 1, 2021 10:02 AM

### Residuals

- The differences between the actual values of something and the predicted values are called **residuals.**
    - residual = data - fit
    - data = fit + residual
    - The residuals are the leftover variation in the data that the model (fit) can't account for.
- We need a criterion to decide what line fits the data the best.
- Observations above regression line are positive residuals and observations below are negative residuals. Our goal is for these residuals to be as small as possible **on average.**

![[Linear Reg/Screen_Shot.png]]

## Least Squares

- The smaller the residuals are, the better the line describes the relationship in the data.
- The most popular approach is to minimize the sum of the squared residuals (squared to account for some being negative and some positive). The resulting line is called the **least squares line.**
    - We want to minimize $\sum e_i^2 = \sum (y_i - \hat{y}_i)^2$ where $e_i = \text{observed} - \text{predicted} = y_i - \hat{y}_i$
    

![[Linear Reg/Screen_Shot 1.png]]

- $s_y$ is the standard deviation of the $y$
- $s_x$ is the standard deviation of the $x$
- The least squares regression line has the following two properties:
    
    ![[Linear Reg/Screen_Shot 2.png]]
    

You use `lm` (stands for linear model) to compute the least squares line in R. 

![[Linear Reg/Screen_Shot 3.png]]

- Here `salePrice` is Y and `livingArea` is X.

### Meaning

- The slope describes the estimated difference in the $y$ variable if the explanatory $x$ for $a$ case happened to be one unit larger.
- The y-intercept describes the average outcome of $y$ when $x = 0$ IF the linear model is valid all the way to $x=0$, which is often not the case.
- If a residual is negative, the model overestimated.
- We should only use the model to make predictions for values that lie within the range of known values. Predicting outside the range is called **extrapolation. You shouldn't extrapolate.**

# $R^2$

- Earlier we evaluated the strength of the linear relationship between two variables earlier using the correlation, $r$.
- It is more common to explain the strength of a linear using using the square of $r$, denoted by $R^2$ (always [0, 1]).
- $R^2$ is the amount of variation in the response variable that is explained by the least-squares regression line.

![[Linear Reg/Screen_Shot 4.png]]

<aside>
💡 The $R^2$ value is a measure of how good the linear model is. The closer $R^2$ is to 100%, the better.

</aside>

# Outliers

- Outliers in linear regression are observations that fall far from the "cloud" of points. They are especially important because they can have a strong influence on the least squares line.
- **Leverage point:** a data point has high leverage if it has particularly high or low predictor $x$ values. These points may strongly influence the slope of the least squares line.
- **Influential points:** if one of these high leverage points appears to actually invoke its influence on least squares line, then we call it an influential point.
- Don't remove outliers without good reason. Clearly indicate any decisions you made when you fit your model so others can understand your decisions.

### Examples

![[Linear Reg/Screen_Shot 5.png]]

Above we add an outlier that is unusual in its $y$ value, but not its $x$ value. It doesn't have a large affect on the line.

![[Linear Reg/Screen_Shot 6.png]]

Above is an example of a leverage point that isn't an influential point.

![[Linear Reg/Screen_Shot 7.png]]

Above is an example of a point with an unusual $x$ value, but a normal $y$ value. It is a leverage point and is an influential point.

### Example: stopping distances