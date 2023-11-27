---
tags: [flashcards, eecs498-dl4cv, eecs445]
source: https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
summary: A loss function quantifies how well our predicted value matches the true value.
---

> [!NOTE] You want a loss function that accurately measures how well your final model is. You don't want to rely on other signals (ex. early stopping or validation loss) because then your model won't be optimized efficiently.

If we try to use a linear classifier, it will fail if the data is not perfectly separable. In this case, we want to find a separator that does the best job of separating the data. We can do this by minimizing empirical risk.

**Loss Function:** a loss or risk function, $\mathcal{L}(\hat{y}^i, y^i)$, quantifies how well (more accurately, how badly) ==$\hat{y}$ approximates y==. smaller values of $\mathcal{L}(\hat{y}^i, y^i)$ indicate that $\hat{y}$ is a good approximation of $y$
<!--SR:!2024-01-29,519,330-->

- Low loss = good classifier
- High loss = bad classifier
- aka **objective function, cost function**
- Negative loss function called **reward function, profit function, utility function, fitness function,** etc.

**Empirical risk** is the average loss over the data points. $\mathcal{L} = \frac{1}{n}\sum_{i=1}^n \mathcal{L}(\hat{y}^i, y^i)$

> [!note]
> You can use different loss functions when calculating empirical risk.
> 

> [!NOTE] Regression vs. classification
> Regression is for prediction continious values (housing prices) while classification is for predicting binned values. Binary classification is used for yes/no types of problems (ex. hot dog or not hot dog) while categorical (aka multi-class) classification is used when there are multiple possible labels (ex. is the image a dog, cat, cow, etc.). 

# Regression Loss Functions
Prediction continious values.

#### [[Mean Squared Error|MSE]]
![[Mean Squared Error|MSE]]

#### [[Regularization|L2 Loss]], [[Regularization|L1 Loss]], and [[L0 Norm]]
The $L^p$ norm is defined as:
$$\|x\|_p=\sqrt[p]{(\left|x_1\right|^p+\left|x_2\right|^p+\cdots+\left|x_n\right|^p)}$$
where each value is raised to the $p$th power, you sum up the individual values, and then take the $p$th root. The L0 norm is $p=0$, L1 is $p=1$, and L2 is $p=2$.

L1 has a sparsity constraint, which drives many of the entries to be identically zero. L2 will prefer them to be small, but non-zero (wants to spread them out). The L0 norm is just the number of non-zero elements so using an L0 loss will force some elements to zero but allow other elements to be large values.

Note that the norms do not divide by the number of examples.

Note that L0, L1, and L2 are norms and not a loss by themselves. They are called a "loss" when they are used in a loss function to measure a distance between two vectors, (ex. $\lVert y_1 - y_2 \rVert_2^2$ ). They can also be used as regularization on the model weights (ex. $\lVert \theta \rVert_2^2$) which are additional terms added to the loss function.

You typically prefer L1 for non-continous data. Ex: if you had a score 1-10, the L1 distance between the scores 3 and 5 is $(5 - 3) = 2$. The L2 norm is $\sqrt{(5^2 - 3^2)} = 2$. If you had a score of  3 and 6, the L1 norm would be 3. The L2 norm would be $\sqrt{(6^2 - 3^2)} = \sqrt{27} \approx 5.19$. This shows that a increase in score of 1 discrete value corresponds to an increase in the L1 norm of 1, but a difference of 3.19 for the L2 norm.

[SO post with some more info](https://datascience.stackexchange.com/a/47111/70970)

# Binary Classification Loss Functions
![[binary-loss-functions-graph.png]]
#### [[0-1 Loss]]
This loss function is 0 for incorrect predictions or 1 for correct predictions. The loss function isn't ==differentiable==, so you can't use 0-1 loss with gradient descent. You can use it with other algorithms like k-Means clustering.
![[0-1-loss-graph.png.png]]
<!--SR:!2025-04-20,554,338-->

#### [[Cross Entropy Loss#Binary Cross-Entropy Loss (aka Log Loss)|Binary-Cross Entropy Loss]]
![[bce-loss.png]]
You use binary cross entropy loss (`torch.nn.BCELoss`) for classification problems where you have a binary (i.e., yes/no) label for each example. For instance, if you were predicting hot dog/no hot dog you could use this. It penalizes the model more when it is confidently wrong.

The inputs to BCELoss should be between `[0, 1]` which can be achieved by applying a [[Sigmoid]] to your model outputs.

#### [[Hinge Loss]]
![[hinge-loss-graph.png]]

#### [[Hinge Loss#Squared Hinge Loss|Squared Hinge Loss]]
![[squared-hinge-loss.png]]


# Categorical (multi-class) Classification Loss Functions
#### [[Cross Entropy Loss]]
Categorical cross entropy loss (`torch.nn.CrossEntropyLoss`) is used when you are predicting over multiple labels for an example (ex. classifying an image as a dog, cat, bird, cow, etc.). It boils down to a combination of [[Softmax]] and [[Cross Entropy Loss#Binary Cross-Entropy Loss (aka Log Loss)|Binary-Cross Entropy Loss]] loss.

Cross Entropy Loss is calculated between the predicted probabilities $S(Y)$ and the actual one-hot vector of labels $L$.
$$
D(S, L) = - \sum_i L_i \log(S_i)
$$
Since $L$ is a one-hot vector, it is filled with zeroes except for a single entry that is 1. We don't want to take the $\log$ of $0$, so we use $L_i$ (serving as the indicator function). Since, $L_i$ can only be $1$ once, this is equivalent to calculating $-\log(S_j)$ where $j$ is the correct label. This is equivalent to calculating BCELoss when the ground truth is 1. Note that $S$ are the **softmaxes of your prediction**.

#### [[F1 Score|Dice Loss]] (aka F1 Score)
For segmentation problems it's easiest to consider the formulation $\frac{2 T P}{2 T P+F P+F N}$. If we were trying to segment a cat in an image that had a cat in the foreground and the remainder of the image were background, we would have the following:
- TP: correctly predicted cat pixels
- FP: background predicted as cat
- FN: cat predicted as background

The advantage of using dice loss is that **it can handle the class imbalance in terms of pixel count for foreground and background**. Below describes why this is:

![[F1 Score#Why does Dice Loss handle class imbalances better than cross entropy loss?#Explanation 1]]

#### [[Focal Loss]]
**Focal loss** adapts the standard CE to deal with extreme foreground-background class imbalance, where the loss assigned to well-classified examples is reduced.

![[Focal Loss#TLDR]]

