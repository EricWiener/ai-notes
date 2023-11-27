Tags: February 15, 2021 12:44 PM

You often have results from a study and want to see if there is a statistical difference between two groups. For example, if you surveyed 1000 United States people and 1000 Germans and found that 50% of the US was stupid and 30% of Germans were stupid, you would want to see whether there was a statistical difference between the groups of if it were due to chance.

We can use computer simulations to try to figure out if there is a meaningful difference. Specifically, you want to know if this difference will likely be observable in the population from which the sample is drawn. **Using sample data to infer something about a population is called statistical inference.**

## Example 1: Pass or fail?

![[Randomizat/Screen_Shot.png]]

- Population: all students at the university who take this course
- Sample: 120 students from last year's offering of the course
- The null hypothesis is that there is no correlation between where a student grew up and whether they pass an engineering course. The variables background and course success are independent. (We assume the boring case for the null hypothesis).
- If the variables are independent, we would expect the proportions $p_{\text{urban}}$ and $p_{\text{rural}}$ to be the same.

![[Randomizat/Screen_Shot 1.png]]

**Data gathered:**

![[Randomizat/Screen_Shot 2.png]]

![[Randomizat/Screen_Shot 3.png]]

- Here we calculate the difference between the proportions for $p_{\text{urban}}$ and $p_{\text{rural}}$.
- We want to see how unusual this difference is if $H_0$ is true.

**Method:**

![[Randomizat/Screen_Shot 4.png]]

- You can remove the data from the cells of the table, but leave the margins.
- You then randomly generate data (with the assumption $H_0$ is true) and try to see what the difference in proportions is.
- If you were doing this with cards:
    - Write pass on 82 cards and fail on 38 cards. Shuffle the cards.
    - Then you deal 65 cards in one pile to represent urban/suburban. You deal 55 cards to represent rural/small-town.
    

**Example trial data:**

![[Randomizat/Screen_Shot 5.png]]

![[Randomizat/Screen_Shot 6.png]]

![[Randomizat/Screen_Shot 7.png]]

- Above shows the results of two trial runs.
- Neither of the differences between populations is as unusual as the observed difference of 25.5%
- If the two variables are independent, we would expect the difference $\hat{p}_{\text{urban}} - \hat{p}_{\text{rural}}$ to be 0 (with some deviation due to sampling error).

**R Simulation**

![[Randomizat/Screen_Shot 8.png]]

- Here we use R to run 10,000 rounds of simulation
- The last line of code `mean(phat_diff > .255)` tells us the percentage of the sample proportion differences $\hat{p}_{\text{urban}} - \hat{p}_{\text{rural}}$ are greater than $25.5\%$. For this experiment, the result is 0.008. Therefore, the sample result is extremely rare if the null hypothesis is true. We are therefore inclined to reject the null hypothesis.

## Example 2: International Cooperation

![[Randomizat/Screen_Shot 9.png]]

- Let $p_c$ represent the proportion of all "suddenly become a CEO and lets their friend win $100,000" stats students. Let $p_s$ represent the proportion of all "suddenly become a senator and let their friend win $100,000" stats students.

![[Randomizat/Screen_Shot 10.png]]

- We just care if the proportions are dramatically different. We don't care if there are a lot more CEO students or a lot more senator students. This will require the two-tailed hypothesis test.

**Example sample data**

![[Randomizat/Screen_Shot 11.png]]

![[Randomizat/Screen_Shot 12.png]]

- Remember $\hat{p}_c, \hat{p}_s$ denote sample proportions

**Method:**

![[Randomizat/Screen_Shot 13.png]]

![[Randomizat/Screen_Shot 14.png]]

**Example Simulation:**

![[Randomizat/Screen_Shot 15.png]]

![[Randomizat/Screen_Shot 16.png]]

- Here the difference between proportions is $-0.1$

**R simulation:**

![[Randomizat/Screen_Shot 17.png]]

- We expect the average difference in the proportions to be about 0.
- We check the average difference between the proportions using `mean(phat_diff >= .2 | phat_diff <= -.2)`. This gives a result of 0.2406, which means a random shuffle produces a difference more unusual than 0.20 on either side of zero.
- Therefore, the estimated p-value for this test is 0.2406. There is little evidence to reject the null hypothesis.

## Example 3:

![[Randomizat/Screen_Shot 18.png]]