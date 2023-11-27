Tags: February 8, 2021 7:45 AM

# Formulating hypothesis statements

You typically have two competing claims:

- The chance model (ex. dolphin was guessing)
- The "something other than chance" model (ex. dolphin was choosing correct button on purpose)

These are called the null hypothesis and alternate hypothesis, respectively.

### Null hypothesis

The **null hypothesis,** denoted by $H_0$, is a statement that usually assumes the "status quo." There is no change, nothing is happening, no difference, no relationship, or no effect in the underlying population.

Typically, it is the claim that any differences we see in the sample results (when compared to the status quo) are due to to chance alone (naturally occurring variability).

### Alternative hypothesis

![[Hypothesis/Screen_Shot.png]]

- The null and alternative hypotheses should not be true at the same time, so if the alternative is true, the null is false, and vice versa.
- The null and alternative hypotheses are statements about a population and not the results in the sample.

### Dolphin Example

![[Hypothesis/Screen_Shot 1.png]]

### Rock-paper-scissors example

![[Hypothesis/Screen_Shot 2.png]]

# Elements of hypothesis testing

A hypothesis testing problem usually includes the following steps:

- Formulate research question and hypothesis
- Collect data
- Analyze the data
- Determine how likely is the data if the null hypothesis is true
- Make conclusions

### Example: US court system

![[Hypothesis/Screen_Shot 3.png]]

![[Hypothesis/Screen_Shot 4.png]]

![[Hypothesis/Screen_Shot 5.png]]

- **Note:** a "not guilty" verdict does not mean the defendant is innocent. It means there was not enough evidence to convince the jury or judge the defendant is guilty.
- In hypothesis testing, we either **reject or fail to reject the null hypothesis.**

<aside>
💡 We do not say we reject or not reject the alternative hypothesis

</aside>

# P-value of a test

![[Hypothesis/Screen_Shot 6.png]]

- In both the examples we did, we simulated data.
- The parameters of interest are $p$, the population proportion.
- In each case, we calculate the percentage of the simulated sample proportions that were **at least as extreme in the direction of the alternative hypothesis** as the observed sample proportion.

### P-value

- When you calculate the proportion of the simulated statistics that are at least as extreme, you are calculating an estimated $p$-value.
- The $p$-value is defined as the probability of obtaining a value of the statistic at least as extreme as the observed statistic when the null hypothesis is true.
- You can estimate it by finding the proportion of the simuylated statistics under $H_0$ that are at least as extreme (in the direction of the alternative hypothessi) as the value of the satistic actually observed in the research study.
- The more repetition you perform, the more accurate the estimate.

![[Hypothesis/Screen_Shot 7.png]]

### Interpreting the p-value

We use the p-values to help us evaluate how well the null hypothesis model explains or "fits" our observed sample results.

- If the p-value is not so small, this indicates that our observed results look like they could be a result of the natural variation that we expect to see when we take random samples.
- The smaller a p-value is, the less inclined we are to think that our sample results are simply due to natural variation.
- Small p-values give us reason to doubt that the null model is a good explanation for our observed results.

### What is a large or small p-value

![[Quick Copy/Screen_Shot.png]]

- Rules are subjective for what values of p are large or small
- The above is a table that provides a guideline, but isn't absolute

![[Interpreting the p-value for the dolphin and rock-paper-scissors examples]]

Interpreting the p-value for the dolphin and rock-paper-scissors examples

### Why not just pick one value as a cutoff?

![[Hypothesis/Screen_Shot 9.png]]

### Summary

![[Hypothesis/Screen_Shot 10.png]]