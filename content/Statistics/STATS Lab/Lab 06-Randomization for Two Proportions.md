Files: lab06-slides-async.pdf

## Exam Prep

- Exam content: Chapter 1, Sections 5.1 - 5.4, 2.1 - 2.4
- Odd practice problems in the textbook
- Make a study guide
- Take the practice exam

# Example 1: Dishwashers

![[Lab 06 Ran/Screen_Shot.png]]

### Defining Hypotheses

- Here we are comparing two samples (neither is a population parameter)
- We have the results of two samples, but we want to generalize to the population.

$H_0$: $p_A - p_B = 0$ (this is equality)

$H_A$: $p_A - p_B < 0$ (this says the rate for A is less than the rate for B).

- $p_A$is the proportion of brand A dishwashers that required a service call in the past year; similarly for $p_B$

### Setting up the simulation

- **Group A**: brand A dishwashers
- **Group B:** brand B dishwashers
- **Succes:** a dishwasher required a service call (a success is defined in terms of the data we look at - not if it is good or bad).
- **Fail:** a dishwasher didn't require a service call
- Result Table
    
    ![[Lab 06 Ran/Screen_Shot 1.png]]
    

**Observed difference in sample proportions of success between Group A and Group B:**

```cpp
(198/600) - (80/200) = -0.07
```

- Note $\hat{p}_A = 198/600$ and $\hat{p}_B = 80/200$

# Making a Data Frame

**You first need to create a vector representing the groups:**

![[Lab 06 Ran/Screen_Shot 2.png]]

- This has 600 for "Group A" and 200 for "Group B"

**We now need to assign results for each of the elements**

![[Lab 06 Ran/Screen_Shot 3.png]]

- We assign 402 failures and 198 successes to Group A.
- We assign 120 failures and 80 successes to Group B.
- Notice that we are trying to reproduce the following table and R puts everything in alphabetical order. In order to replicate this, we need to also put things in alphabetical order, so we put "Failure" above "Success" because "F" comes before "S."
    
    ![[Lab 06 Ran/Screen_Shot 1.png]]
    

**We can then create the Data Frame**

![[Lab 06 Ran/Screen_Shot 4.png]]

- We assign the `groupRepeat` vector to the "group" variable and `resultRepeat` to the "result" variable.
- Use `stringsAsFactors = TRUE` so R treats the data as categorical variables and not text.

**You can use `addmargins()` to add a "Sum" row and column**

![[Lab 06 Ran/Screen_Shot 5.png]]

# Computing the Statistic

To find the observed sample proportion of Group A:

- Find the rows that are both Group A and Succcess
- Count up the number of these rows
- Divide by the total in Group A

![[Lab 06 Ran/Screen_Shot 6.png]]

- The `&` is the boolean AND.

You can repeat this for Group B and then do `statistic <- proportion1 - proportion2`

# Run one iteration

![[Lab 06 Ran/Screen_Shot 7.png]]

- You would have 522 yellow for Failure and 278 blue for Success. You then shuffle these up and deal 600 to Group A and 200 to Group B.
- This would tell you how many each group would get if they both had the same probability of a Success/Failure.

**We can run one iteration with the function they wrote for us:**

```cpp
# Takes in a data frame and outputs data frame
shuffle <- shuffle_two_groups(dishwashers)
```

- This just shuffles the Success/Failure column to distribute the values randomly among Group A and Group B.

**We can then look at the results:**

![[Lab 06 Ran/Screen_Shot 8.png]]

**We can find the difference in the sample proportions like we did before:**

![[Lab 06 Ran/Screen_Shot 9.png]]

**We can then wrap everything in `replicate` to repeat this process 1000 times and create a vector of the simulated differences**

![[Lab 06 Ran/Screen_Shot 10.png]]

**Visualizing results**

![[Lab 06 Ran/Screen_Shot 11.png]]

- We can use a histogram to visualize the results.

## **Computing a p-value**

<aside>
💡 The **p-value** is the probability of obtaining a value of the statistic at least as extreme as the observed statistic when the null hypothesis is true.

</aside>

- $H_0$: $p_A - p_B = 0$
- $H_A$: $p_A - p_B < 0$
- **More extreme** means "closer to the alternative hypothesis".
- We can add up the number of simulationResults less than or equal to our observed $\hat{p}_A - \hat{p}_B$

```r
sum(simulationResults <= statistic) / 1000
# 0.032
```

This p-value is unusual, so, we'll conclude that we have strong evidence in favor of the alternative hypothesis. That is, we have strong evidence to support the claim that Brand A dishwashers have a smaller percentage of service calls when compared to Brand B.