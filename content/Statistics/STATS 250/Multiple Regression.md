Files: multiple_regression.pdf
Tags: April 12, 2021 7:23 AM

You can have linear regression at both the sample and population level.

- Sample level: $y_i = \beta_0 + \beta_1x_i + \epsilon_i$
    - $y_i$ is the response, $x_i$ is the predictor, and $\epsilon_i$ is the error
- At the population level, the **mean response** is: $\mu_{Y|X=x} = \beta_0 + \beta_1 x$
    - If we average all the data, we get the population model
    - Here, we say for a particular value of $x$, what is the average response
    

# Multiple regression

- Extends simple linear regression to the case that still has one response, but many predictors (denoted as $x_1, x_2, ...$). The method is motivated by scenarios where many variables may be be simultaneously connected to an output.

### Categorical predictors

Example: you fit a linear regression model for interest rate with a single predictor variable indicating whether or not a person has a bankruptcy in their record:

![[Multiple R/Screen_Shot.png]]

**Interpret the coefficient for the past bankruptcy variable in the model. Is this coefficient significantly different from 0 ?**

- The variable takes one of two values: 1 when the borrower has a bankruptcy in their history and 0 otherwise.
- A slope of 0.737 means that the model predicts a 0.737% higher interest rate for those borrowers with a bankruptcy in their record.
- Examining the regression output, we can see that the p-value for is very close to zero, indicating there is extremely strong evidence the coefficient is different from zero when using this simple one-predictor model.

### Fitting a linear regression model for a categorical variable with 3 categories

- You can encode 3 levels of one variable using just two extra predictors (not 3).

![[Multiple R/Screen_Shot 1.png]]

![[Equation for this regression model]]

Equation for this regression model

## Many predictors in a model

![[Multiple R/Screen_Shot 3.png]]

![[Multiple R/Screen_Shot 4.png]]

### Example: Many predictors in a model

![[Multiple R/Screen_Shot 5.png]]

**What does $\beta_5$, the coefficient of bankruptcy, represent? What is the point estimate of $\beta_5$? What does this mean?**

$\beta_5$ is a model parameter, which represents the mean change interest rate if someone had a bankruptcy compared to someone who didn’t have a bankruptcy, all other factors held even.

The point estimate is $b_5 = 0.405$. We estimate that, on average, a borrower who has had a bankruptcy will have an interest rate that is 0.405% higher than a borrower who has not had a bankruptcy when all the other predictors have the same values.

**Use the multiple regression model to estimate the interest rate for the first observation in the data set and calculate the residual. Estimated interest rate:**

![[Multiple R/Screen_Shot 6.png]]

## Why does the coefficient change when adding additional variables?

![[Multiple R/Screen_Shot 7.png]]

When we estimated the connection of the outcome interest rate and the predictor bankruptcy using simple linear regression, we were unable to account for the predictive power of other variables like the borrower’s debt-to-income ratio, the term of the loan, and so on. That original model was constructed in a vacuum and did not consider the full context. When we include all of the variables, underlying and unintentional bias that was missed by these other
variables is reduced or eliminated.

If we examined the data carefully, we would see that some predictors are correlated. We say the two predictor variables are collinear (pronounced as co-linear) when they are correlated.

This collinearity complicates model estimation. While it is impossible to prevent collinearity from arising in observational data, experiments are usually designed to prevent predictors from being collinear.

### The y-intercept

In our multiple regression model, the estimated y-intercept is 2.243 – this is the model’s predicted interest rate when each of the variables take value zero: the borrower has a mortgage on their home, the borrower has no debt (debt-to-income and credit utilization are zero), they have not had a bankruptcy, and so on.

Many of the variables do take a value 0 for at least one data point; however, one variable never takes a value of zero: term, which describes the length of the loan, in months. If term is set to zero, then the loan must be paid back immediately; the borrower must give the money back as soon as they receive it, which means it is not a real loan. **Ultimately, the interpretation of the
intercept in this setting is not insightful.**

### Checking conditions for linear regressions (LINE)

- Linearity: The data should show a linear trend. If there is a nonlinear trend, an advanced regression method from another book or later course should be applied.
- Independent observations: Be cautious about applying regression to data collected sequentially in what is called a time series. Such data may have an underlying structure that should be considered in a model and analysis.
- Nearly normal residuals: Generally, the residuals must be nearly normal. When this condition is found to be unreasonable, it is usually because of outliers or concerns about influential points.
- Constant variability: The variability of points around the least squares line remains roughly constant.

![[Multiple R/Screen_Shot 8.png]]

There appears to be a negative relationship between the residuals and the fitted values (we should be seeing relatively random scatter about the horizontal line). The normal Q-Q plot has a definite curve in it making us question the nearly normal condition. What these show us is that we may have omitted an important variable from our model.

### Extra notes

- Remember not try to extrapolate. If a value lies outside the range of the data, you shouldn't try to make a prediction.