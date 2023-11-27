Tags: February 15, 2021 12:44 PM

### Decision Errors

![[Decision E/Screen_Shot.png]]

- Type 1 Error: the null hypothesis ($H_0$) is true, but we falsely reject it. **We reject** $H_0$.
- Type 2 Error: the null hypothesis ($H_0$) is false, but we fail to reject it. **We fail to reject** $H_0$.

<aside>
💡 The 0.05 significance level proposed by Fisher and the FDA exists because they determined the probability of a Type 1 error should be no more than 5%. That is, **we should reject a true null hypothesis only at most 5% of the time.**

</aside>

![[Example using the US justice system.]]

Example using the US justice system.

If you used the example of deciding whether someone is sick, a **false positive** is type 1 error (decide in favor of $H_A$ when $H_0$ is true). If you say a sick person is healthy, this is a **false negative** (decide in favor of $H_0$ when $H_A$ is true).

## Reducing the error rate

- If you want to reduce the Type 1 Errors in the US court system (aka reduce an innocent person being sent to jail), you would need to make it harder to convict someone of a crime.

<aside>
💡 Any time you decrease the Type 1 Error rate you increase the Type 2 Error rate (and vice versa).

</aside>

- Improving your ability to investigate crimes will help with both error rates. In statistics, you can sometimes reduce both error rates by having more data.

## Choosing a significance level

- The chance of making a Type 1 Error is called the "significance level" of the test.
- Often, a researcher will decide what significance level / Type 1 Error rate they are comfortable with, based on the context of the research.

![[Decision E/Screen_Shot 2.png]]

- By choosing a small significance level, you minimize the chance of Type 1 Error, but also increase the chance of Type 2 Error.
- A larger significance level reduces the chance fo Type 2 Error, but increases the chance of Type 1 Error.
- In the real world, the significance levels should reflect the real consequences.
- It is standard practice to set up $H_0$ and $H_A$ so that the Type 1 Error is the more important.

# Hypothesis Tests

- Sometimes you only consider one tail and sometimes you consider two tails.

![[Decision E/Screen_Shot 3.png]]

![[Decision E/Screen_Shot 4.png]]

- You care if there is a preference, so you want the hypothesis test to account for a majority choosing to be senators or a minority choosing to be senators.

## Caution:

- Hypotheses should be set up before seeing the data. For example. if you originally have a two-sided hypothesis test and then switch to one-sided after seeing the results, it's bad practice.
- For a two-sided test, you often take the area in a single tail and double it to get the p-value.