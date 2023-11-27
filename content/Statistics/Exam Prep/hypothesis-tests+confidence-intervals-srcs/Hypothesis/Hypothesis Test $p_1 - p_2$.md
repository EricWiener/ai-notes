Created: March 29, 2021 11:20 AM

## Conditions for $p_1 - p_2$

![[Exam 2 Rev/Screen_Shot 11.png]]

- Here, we use the **pooled estimate of the sample proportion** as the proportion when checking if the samples are large enough.
- You just add up total successes and divide by total sample size.
- We run a hypothesis test assuming $H_0$ is true until we can't assume it anymore, so we check conditions assuming $H_0$ is true.
    - $H_0$ says $p_1 = p_2$. The best we can do is get a single $\hat{p}$ that combines data from both groups.

### Conditions - long form

1. **Independence within each sample**: The observations within each sample
are independent (e.g., we have a random sample from each of the two
populations).
2. **Independence between the samples**: The two samples are independent of
one another such that observations in one sample tell us nothing about
the observations in the other sample (and vice versa).
3. We have a sufficiently large sample size, meaning ALL of the following are true:
    - $n_1\hat{p} \geq 10$
    - $n_1(1 - \hat{p}) \geq 10$
    - $n_2\hat{p} \geq 10$
    - $n_2(1 - \hat{p}) \geq 10$
    - **Where $\hat{p}$ is the pooled estimate of the sample proportion**
    
- For copy-pasting on exam
    
    n1(phat) >= 10
    n1(1 - phat) >= 10
    n2(phat) >= 10
    n2(1 - phat) >= 10
    

![[Lab 08 Inf/Screen_Shot 12.png]]

- You now check with `pHatPooled`

### Run hypothesis test

**Setup question**

![[Lab 08 Inf/Screen_Shot 11.png]]

![[Lab 08 Inf/Screen_Shot 13.png]]

- This p-value tells us likelihood $p_1 - p_2 > 0$ (note that we use the population proportions here - not the sample proportions).
- Ignore confidence interval here (we did a one-sided test but it should have been two-sided).