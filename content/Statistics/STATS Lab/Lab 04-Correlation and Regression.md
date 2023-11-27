![[Lab 04 Cor/Screen_Shot.png]]

- x is the explanatory variable
- y is the response variable
- There is a positive, decently linear relationship with some clustering.

# The `~` operator.

- Read as "by" or "versus".

![[Lab 04 Cor/Screen_Shot 1.png]]

- You can use this to show the distribution of penguin body mass **by** penguin species.
- Previously we had to explicitly create the subsets:
    
    ![[Lab 04 Cor/Screen_Shot 2.png]]
    
- 

## ~ in scatterplots

![[Lab 04 Cor/Screen_Shot 3.png]]

- Note this is **Y vs. X**
- You don't need `penguins$` prefix on each of the variables because he added the `data = penguins` argument.

![[Lab 04 Cor/Screen_Shot 4.png]]

# Strength and Correlation

- Quantify strength of a **linear relationship** using $r$.
    - Note: it says nothing about strength of a non-linear relationship
    - Also note that a horizontal line is still a perfect line.
- As $|r|$ approaches 1, the linear relationship gets stronger.

Use the `cor()` function (returns a scalar):

![[Lab 04 Cor/Screen_Shot 5.png]]

## Correlation matrix

- A correlation matrix tells you the correlation between multiple pair of variables.
- In order to do this, you need to use the `select` argument to `subset()` to choose which variable to keep.
- You pass the `select` a vector of variable names using `c()`.
- You don't keep categorical variables - only numeric.

![[Lab 04 Cor/Screen_Shot 6.png]]

![[Lab 04 Cor/Screen_Shot 7.png]]

- Each entry in `cor` is the correlation between the corresponding row and column variable.
- This is a symmetric matrix, so it doesn't order what order you look at variables.

# Linear regression

- This will find the line as close to as many points as it can.

![[Lab 04 Cor/Screen_Shot 8.png]]

- `lm` stands for linear model.

```cpp
summary(reg1)
```

![[Lab 04 Cor/Screen_Shot 9.png]]

- You can then summarize the data using `summary`.
- Note if you just run `lm` and don't store it into a variable, it won't print out as much information as it does when you save it into a variable and use `summary`.
- The equation for a regression line is $\hat{y} = b_0 + b_1 x$
- $b_0, b_1$ are sample estimates of the true population parameters.
- Note that $b_1$ will always be in the row corresponding to the explanatory variable (`bill_length_mm` here).
- `S` is the residual standard error. This is a measure of the variability in the residuals that's left over after we fit the regression line.
- We will be using "Multiple R-squared" not "Adjusted R-squared". Multiple R-squared is just the correlation squared. This is a measure of how much of the variability in the response variable can be explained with the explanatory variable (ex. 34.75% of the variability in body mass can be explained by the linear relationship with bill length).

## Plotting a regression line

![[Lab 04 Cor/Screen_Shot 10.png]]

- You use `abline` to plot the regression line on the scatterplot.