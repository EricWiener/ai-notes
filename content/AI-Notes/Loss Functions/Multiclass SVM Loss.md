---
summary: aims to get the score for the correct class higher than all the other scores
---
This loss function aims to get the score for the correct class higher than all the other scores. It recieved it's name because it is commonly used when training SVMs. However, it can be used with any type of model. Also important to note that an SVM with a linear kernel is just a linear classifier.

![[multi-class-loss-graph.png.png]]

- In the diagram above, the bottom axis is the score for a class. The vertical axis is the loss value.
- The dash on the bottom axis represents the highest score predicted for any class besides the correct class
- We want to ensure there is a sufficient margin between the score predicted for the correct class and the highest score predicted for any of the incorrect classes
- If the margin is large enough, the loss is zero. Otherwise. the loss increases inversely proportional to the margin size.

### Math

- Given an example $(x_i, y_i)$ - $x_i$ would be the training example and $y_i$ is the label
- Let $s = f(x_i, W)$ be the scores

Then the SVM loss for a particular training example has the form:

$$
 ⁍
$$

- $\sum_{j \neq y_i}$: You sum over all the incorrect labels for a certain training example
- $\max(0, s_j - s_{y_i} + 1)$: you take the difference between the score for the incorrect label and the score for the correct label.
    - In our case, we are using $+1$ as a margin, which will ensure the correct score is at least $1$ greater than the incorrect score.
    - If the score is negative, this means there is a sufficient gap between the scores (the score for the correct label is greater than the score for the incorrect label), so the loss becomes 0.
    - Otherwise, the loss is equal to the difference between the scores.

### Example

![[The red values at the bottom are the SVM loss values for each class.]]

The red values at the bottom are the SVM loss values for each class.

Here we have the raw scores given by the model for each of the classes (for three training examples).

The loss for the cat can be calculated as:

$= \max(0, 5.1 - 3.2 + 1) + \max(0, -1.7 - 3.2 + 1) \\ = \max(0, 2.9) + \max(0, -3.9) \\ = 2.9 + 0 = 2.9$

Because the gap between the score for the cat (correct label) and the frog (incorrect label) was large enough (greater than our margin $1$), the loss for that pair was 0.

- Note: you only sum over the incorrect labels

We repeat the same process for all training points. To get the overall loss for all training examples, we just **average** the losses for each training point. $L = (2.9 + 0.0 + 12.9) / 3 = 5.27$

### Questions

**What happens to the loss if the scores for the car change a bit?**

The car image was correctly classified and previously had a loss of 0. Our loss would not change at all because the difference between the score for the correct label and the incorrect label was over the margin. This loss is robust to small changes in scores.

**What are the min and max possible values?**

The min is zero and the max is infinity. However, here a minimum loss of zero is achievable, while it wasn't achievable with Cross-Entropy loss with soft-max.

**If all the scores were random, what loss would we expect?**

It would be approximately $C - 1$ where $C$ is the number of classes 

- This is because the loss for a particular training image is $L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + 1)$
- We sum over everything but the correct class, so we sum over $C - 1$ items (minus one because we don't include the correct labelled class in the summation).
- Since all scores are random, the correct score $s_{y_i}$ and the score for another class, $s_j$, would be about the same. Therefore, $\max(0, s_j - s_{y_i} + 1)$ just becomes $\max(0, \approx 1) = 1$.
- Since we sum over $C - 1$ items, we do $(C - 1)(1) = C - 1$.

**Is SVM Loss differentiable?**

It is differentiable everywhere but the hinge, but the probability we will hit a point exactly on the hinge is almost 0, so it's okay to use it in practice. There is also a notion of **sub-differentiability**, which it has.

**How does changing the margin affect the problem?**

If we change the margin from $1$ to $0.1$, but scale up all the prediction scores by a factor of 10, we will get the exact same thing. Therefore, it is common practice just to leave it at 1. There is no need to do a hyper-parameter search for the best margin.

## Cross-Entropy Loss vs SVM Loss

The choice of loss function expresses to the classifier that we have different preferences for the types of scores we want to recieve.

**Cross-Entropy Loss:** $L_i = -\log(\frac{\exp(s_{y_i})}{\sum_j \exp(s_j)})$

**SVM loss:** $L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + 1)$

Assume we had the scores:

$\begin{aligned}
&{[10,-2,3]} \\
&{[10,9,9]} \\
&{[10,-100,-100]}
\end{aligned}$

and $y_i = 0$ (the correct label is label 0)

- The  SVM loss would be $0$ because the difference between the correct score and the other scores is $\geq 1$. This means we are perfectly happy with these scores.
- The cross-entropy loss would be greater than 0. It is always non-zero because the logits are never exactly 0.

**What happens to each loss if you slightly change the scores of the last datapoint (**$[10,-100,-100]$)**?**

- The cross-entropy loss will change because you always get a non-zero loss, so it can always be improved. **Cross-entropy will always try to drive the scores apart.**
- SVM loss is already giving zero loss, so it won't change. **Once the difference between the correct score and incorrect scores is large enough, it doesn't care.**

> [!note]
> Cross entropy will always try to drive apart the scores for the correct and incorrect classes. For SVM loss, once the difference is greater than the margin, it will no longer cause changes.
> 