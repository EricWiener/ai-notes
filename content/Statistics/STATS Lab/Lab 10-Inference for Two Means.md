Files: lab10-slides-async.pdf

### Conditions for one mean (revisited)

In order to use the $t$ distribution for constructing confidence intervals and performing hypothesis tests for means, we need two conditions to hold.

1. The observations within the sample are independent.
2. The observations are from a population with a nearly normal distribution

## Conditions for two means

What happens if we end up collecting data from 2 numeric variables? How does this affect our conditions for inference?

![[Lab 10 Inf/Screen_Shot.png]]

# Independent or not

![[Lab 10 Inf/Screen_Shot 1.png]]

# Name the parameter

![[Lab 10 Inf/Screen_Shot 2.png]]

![[Lab 10 Inf/Screen_Shot 3.png]]

# R Code

### Checking if a distribution is bimodal

We can use a boxplot that shows `flipper_length_mm` by species.

```r
boxplot(flipper_length_mm ~ species, data = penguins, main="Flipper length by species", xlab = "Species", ylab = "Flipper length (mm)"
```

![[Lab 10 Inf/Screen_Shot 4.png]]

- This boxplot shows us that Gentoo penguins have longer flippers
- We can just focus on Adelie and Chinstrap penguins (since Gentoo are probably different)

# Hypotheses for the Difference of Two Population Means

$H_0: \mu_1 - \mu_2 = 0 \\ H_a: \mu_1 - \mu_2 \neq 0$

Where $\mu_1$ is the mean flipper length in mm for Adelie penguins on Palmer Archipelago, and $\mu_2$ is the mean flipper length in mm for Chinstrap penguins on Palmer Archipelago

Before we proceed, we'll subset the data to just involve Adelie and Chinstrap penguins. This is only because we're not interested in Gentoo penguins for this question, so we'll take them out.

### Subset

```r
# Only keeps penguins whose species is Adelie or Chinstrap
penguinsSubset <- subset(penguins, species %in% c("Adelie", "Chinstrap"))

```

### Conditions for the Difference of Two Population Means

In order to use the $t$ distribution to construct confidence intervals and perform hypothesis tests for the difference in two means, we need some conditions to hold. **What are they?**

1. *Condition 1*
a. independence within each group
b. independence BETWEEN each group
2. *Condition 2*
a. observations of flipper length from Adelie penguins come from a nearly-normal population
b. observations of flipper length from Chinstrap penguins are from a nearly-normal population

Let's check these conditions!

1. *Condition 1*
a. Adelie penguins and Chinstrap penguins are randomly sampled (not sure here, proceed with caution)
b. Adelie penguins are separate from Chinstrap penguins because they are different species, so they must be independent of one another
2. *Condition 2*
a. make a histogram of flipper length of sampled Adelie penguins, and look for a bell-shaped distribution
    
    ![[Lab 10 Inf/Screen_Shot 5.png]]
    
    b. make a histogram of flipper length of sampled Chinstrap penguins, and look for a bell-shaped distribution
    
    ![[Lab 10 Inf/Screen_Shot 6.png]]
    

## t-test for Difference in 2 Population Means

```r
# this is y vs. x (comparing flipper length by species)
# this is <quantitative variable> ~ <categorical variable>
t.test(flipper_length_mm ~ species,
       data = penguinsSubset,
       mu = 0, # mu_1 - mu_2 = 0
       alternative = "two.sided")
```

![[Lab 10 Inf/Screen_Shot 7.png]]

We have _______________ (very strong / strong / some / little) evidence to suggest that there ______ (is/is not) a difference in the ____________ (mean/proportion) flipper length for the two groups, Adelie penguins and Chinstrap penguins.

We have **very strong** evidence to suggest that there **is** a difference in the **mean** flipper length for the two groups, Adelie penguins and Chinstrap penguins.

- Conclusions are about the **alternative** hypothesis
- We should **not** imply that the null is true or false.
- Avoid being too strong: we *have evidence for* the alternative.

# Confidence Intervals for Difference in two Population Means

Parameter is $\mu_1 - \mu_2$

### Conditions for the Difference of Two Populations Means

The good news here is that the conditions are identical for a confidence interval and a hypothesis test. So we have already verified any necessary conditions.

### Computing 90% confidence interval

```r
t.test(flipper_length_mm ~ species,
       data = penguinsSubset,
       conf.level = 0.9) # Defaults to two.sided
```

![[Lab 10 Inf/Screen_Shot 8.png]]

Notice that our 90% confidence interval has both *negative* values. What does this tell us about our data?

We estimate with 90% confidence that the true flipper lengths for Adelie penguins are between 7.411 mm and 4.031 mm **shorter** than Chinstrap penguins, on average.