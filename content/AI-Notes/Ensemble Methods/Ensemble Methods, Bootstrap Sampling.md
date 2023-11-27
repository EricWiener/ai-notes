---
tags: [flashcards]
source: https://statisticsbyjim.com/hypothesis-testing/bootstrapping/
summary: Ensemble methods combine multiple models to get better results. Bootstrap sampling creates random subsets of the original dataset (of the same size).
---

Goal: decrease variance (estimation error) without increasing bias (structural error)
Idea: average multiple models to reduce variance

### Decision Trees Ensemble Methods:
- If we keep running the decision tree algorithm on the same dataset, you’ll get the same tree each time. Doesn’t help.
- If you just split the dataset into thirds, it isn’t necessarily balanced. Could get unbalanced datasets.

### Bootstrap Sampling: selecting random subsets of dataset
![[AI-Notes/Ensemble Methods/ensemble-methods-bootstrap-sampling-srcs/Untitled.png]]
[Source](https://www.kaggle.com/prashant111/bagging-vs-boosting)

- **Bootstrap** refers to random sampling with replacement (you keep sampling from the same original dataset - you don't remove data once it's been sampled). Bootstrap allows us to better understand the bias and the variance with the dataset.
- So, **Bootstrapping** is a sampling technique in which we create subsets of observations from the original dataset with replacement. **The size of the subsets is the same as the size of the original set.**
- **The probability an example is selected to be in a subset is ~63%**
    - Take an *n* sided dice and toss is *n* times. Every-time side i comes up, add that to your training dataset.
        
        ![[AI-Notes/Ensemble Methods/ensemble-methods-bootstrap-sampling-srcs/Untitled 1.png]]
        The top shows the entire dataset. The bottom shows the training examples selected.
        
    **Probability of something being selected:** Note: $e^x = \lim_{n\to\infty} (1 + \frac{x}{n})^n$
    
    Note: $e^x = \lim_{n\to\infty} (1 + \frac{x}{n})^n$
    
    - The probability of selecting a specific training example is $\frac{1}{n}$
    - The probability of not selecting a specific training example is $1 - \frac{1}{n}$
    - The probability of not selecting a specific training for an entire bootstrapped dataset is $(1 - \frac{1}{n})^n =(1 + \frac{-1}{n})^n$
    - This looks like $e^{-1}$ as $n \to \infty$
    - $e^{-1} \approx 36$%. Therefore the probability something isn’t selected is about 36% and the probability it is selected is about 63%.