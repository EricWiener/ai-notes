Tags: January 27, 2021 9:32 AM

- A **summary measure** for a set of quantitate data is a number computes from the data that describes certain aspects of the data.
- The collection of all data that is relevant to a study is called the population for that study.
- Statistics deals with situations where we don't see all the data. Usually we see a sample, which is a sub-collection of data from the population.
- A summary measure computed for a population is called a **parameter.**
- A summary measure computed for a sample is called a **statistic**.
- We use sample statistics to estimate population parameters.

### Mean

- You just add everything up and divide by the number of items

```cpp
x = c(1, 2, 3, 4, 5)
mean(x)
```

- If the data is a population, then the mean is denoted by $\mu$.
- If the data is a sample, then the mean is denoted by $\bar{x}$.
- This is the number that is closest to all of the data
- **Skewed a lot by outliers.**

<aside>
💡 It is typical to round the mean to one more decimal place than given in the data. Don't round the mean to a whole number since it does not represent a possible value of the variable.

</aside>

### Proportion

- A special case of the mean when we have **dichotomous** (two types) data.
- Example: flip a coin 5 times and get 3 heads and 2 tails. The proportion of heads is 3/5.

**Dichotomous/Binary Variables:**
A **dichotomous** variable is a variable that can take on two types.

When we code dichotomous variables with the options as 1s and 0s the variable is called a **binary** variable. In these situations, a ‘1’ indicates that the category we want to count and a ‘0’ indicates the category we are not counting. For example, for the question “What proportion of U.S. adults prefer print books over e-books?” we code ‘1’ if the U.S. adult prefers print books and ‘0’ if they don’t.

### Median

- The midpoint of the data is we order it from smallest to largest.
- If the number of data points is even, it is the average of the middle two points.
- If the number of data points is odd, it is the middle point.
- **Not skewed a lot by outliers.**
- Median is used for income data (so you don't get skewed by super rich).
- Use the same rule for rounding as mean.

```cpp
x = c(1, 2, 3, 4, 5)
median(x)
```

<aside>
💡 Notes: The mean is sensitive to extreme observations (can change dramatically due to a few extreme observations). The median is resistant/robust to extreme observations.

</aside>

### Central tendency

- Both the mean and median describe the **central tendency** of the data.
- The mean is also closely related to the notion of center of mass of an object, which is the point where it can be balanced.
- A measure of central tendency can be thought of as a number that is most similar to most of the data.

# Percentile and interquartile range

- For any number between 0 and 100, the pth percentile is the value such that p% of the observations fall at or below that value.
- The median is the 50th percentile.
- You could have multiple values. If the data is {1, 2, 3, 4, 5, 6}, then the 50th percentile could be any value between 3 and 4. You usually choose the middle value of possible values.
- First quartile (25th percentile), median / second quartile (50th percentile), third quartile (75th percentile).
- To find Q1 and Q3, you compute the median of the halves of the data after you find the median.