---
tags: [flashcards]
aliases: [homoscedastic uncertainty, homoscedastic]
source: https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/homoscedasticity/
summary: a sequence of random variables is homoscedastic if all its random variables have the same finite variance
---
Homoscedasticity (meaning “==same variance==”) describes a situation in which the error term (that is, the “noise” or random disturbance in the relationship between the independent variables and the dependent variable) is the same across all values of the independent variables.  In the context of homoscedastic loss, this means that all examples in your training data will have the same variance between input features and target for a particular task (ex. occupancy or velocity regression).
<!--SR:!2026-11-02,1193,330-->

Heteroscedasticity (the violation of homoscedasticity) is present when the size of the error term differs across values of an independent variable.  The impact of violating the assumption of homoscedasticity is a matter of degree, increasing as heteroscedasticity increases.


A simple bivariate example can help to illustrate heteroscedasticity: Imagine we have data on family income and spending on luxury items.  Using bivariate regression, we use family income to predict luxury spending.  As expected, there is a strong, positive association between income and spending.  Upon examining the residuals we detect a problem – the residuals are very small for low values of family income (almost all families with low incomes don’t spend much on luxury items) while there is great variation in the size of the residuals for wealthier families (some families spend a great deal on luxury items while some are more moderate in their luxury spending).  This situation represents heteroscedasticity because the size of the error varies across values of the independent variable.  Examining a scatterplot of the residuals against the predicted values of the dependent variable would show a classic cone-shaped pattern of heteroscedasticity.