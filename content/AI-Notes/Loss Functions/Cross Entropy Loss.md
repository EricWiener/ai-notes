---
summary: measures the difference between the true probability distribution and the calculated one.
---

**Related:** [[Linear Classifier]]
**Cross Entropy Loss** measures the difference between the true probability distribution and the calculated one. This is needed when we have multiple labels we are predicting (no longer binary classification). It gives us a measure of how well our classification algorithm is doing.

### Binary Cross-Entropy Loss (aka Log Loss)
![[bce-loss.png]]
The diagram above shows BCE loss when y is true on the left and when y is false on the right. The x-axis is the probability predicted that $\hat{y}$ is true. It is higher (worse) when the model is more confidently wrong.

The binary cross entropy loss is given as:

$$
\mathcal{J} = -\sum_{i=1}^m y^{(i)}\log(\hat{y}^{[i]}) + (1 - y^{(i)})\log(1 - \hat{y}^{[i]})
$$

$\hat{y}$ usually represents the logits given by softmax: $\hat{y} = \text{softmax}(\theta)$.

$\mathcal{J} = \text{CE}(y, \hat{y})$.

**Gradient of binary cross-entropy loss:**
$$
\frac{\partial \mathcal{J}}{\partial \hat{y}} = \hat{y} - y
$$

### Categorical Cross-Entropy Loss
This is implemented in PyTorch by `torch.nn.CrossEntropyLoss`. It is used when you can assign multiple labels to a single training example vs. binary cross entropy loss (`torch.nn.BCELoss`) where you have a single label for each example. To give an example, you would use `CrossEntropyLoss`  if you wanted to perform image classification over multiple classes (dog, cat, mouse, etc.). You would use `BCELoss` if you just wanted a yes/no answer (is/isn't a hot dog).

For `torch.nn.CrossEntropyLoss` the target can be a probability distribution over the classes. For `torch.nn.BCELoss` the target is a one-hot vector for each example.

### Multinomial Logistic Classification:
![[cross-entropy-loss-pipeline.png|400]]
[Pipeline for Multinomial Logistic Classification](https://www.youtube.com/watch?v=tRsSi_sqXjI)

The above picture shows the steps needed for Multinomial Logistic Classification (aka a linear classifier for multiple classes).

1. You have an **input** $x$ and compute $Wx + b$ to get raw scores or **logits.**
2. The logits aren't normalizied, so we use the **softmax** function to convert them into probabilities.
3. We then compute the Cross Entropy Loss between the **probability** distribution predicted $S(Y)$ and the actual probability distribution, our one-hot vector of **labels.**
4. We use the Cross-Entropy Loss calculated to update the weights of our model and repeat.

### Math:
Compare $\hat{y}_c$ with the ground truth vector $\bar{y}$ using cross entropy loss. $\mathcal{L}(\bar{y}, \hat{y}_c) = -\sum_c y_c \log \hat{y}_c$

![[compare-gt-with-pred-vec.png|500]]

[Cross Entropy Loss:](https://www.youtube.com/watch?v=tRsSi_sqXjI)

Cross Entropy Loss is calculated between the predicted probabilities $S(Y)$ and the actual one-hot vector of labels $L$.

$$
D(S, L) = - \sum_i L_i \log(S_i)
$$

Since $L$ is a one-hot vector, it is filled with zeroes except for a single entry that is 1. We don't want to take the $\log$ of $0$, so we use $L_i$. Since, $L_i$ can only be $1$ once, this is equivalent to calculating $-\log(S_j)$ where $j$ is the correct label.

If the item is correctly classified, then $S_i$ will be close to $1$ and $-\log(1)= 0$. If the item is incorrectly classified, $S_i$will approach 0 and $-\log(0) \rightarrow \infty$. We are **minimizing** the Cross Entropy Loss, so it makes sense that an incorrect prediction will have a higher loss value, while a better one will have a lower loss value.

**Min/Max Possible Value of** $L_i$
- Min 0: if you predict the perfect 1-hot distribution. However, you can't actually predict 0, so this isn't really obtainable (you'll predict something very close to 0).
- Max: positive infinity. This is because if you predict an extremely small score for the correct class, you will get a value of $-\log(x)$ with a small value of $x$. The smaller $x$ is, the closer this expression is to positive infinity.

**If all scores are small random values, what is the loss?**
The loss would be $-\log(1/C)$ where $C$ is the number of categories, because we expect a uniform prediction over the categories.

![[Softmax#Softmax Function]]

### Numerical Instability
You can run into issues with numerical stability if you do $e^x$ where $x$ is a big number. This can be fixed by shifting the values you are going to $\exp$ so that the largest value is 0.

![[how-does-numerical-stability-work.jpeg|700]]

### Gradient
![[how-to-take-gradient-softmax.jpeg|700]]

### Example
![[example-cross-entropy-calc.png|700]]

In the above example, we have three different values for labels (1, 2, 3). Therefore, our neural network will need to have three different output nodes that each give a probability that the datapoint should be labeled 1, 2, or 3. These outputs are given as logits and in our particular example, the logits are [2, 4, 0.5]. We need to use the soft-max on these logits to convert them to probabilities. Then we can calculate the cross-entropy loss.