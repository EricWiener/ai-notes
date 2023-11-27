---
tags: [flashcards, eecs445]
source:
summary: Simplest linear classifier. If the data isn’t separable, it won’t converge and isn’t guaranteed to find the separator with the smallest training error.
---

## Linear Classifier
$h(\bar{x}, \bar{\theta}) = \{+1, -1\}$: possible labels = $sign(\bar{x} \cdot \bar{\theta})$
$\bar{x} \in \mathbb{R}^d$: feature vectors
$\bar{\theta}$: your parameters
**The choice of** $\bar{\theta}$ **determines:**
1. Orientation of the hyperplane
2. The class label on either side of the hyperplane
**Goal**: Want to have the best chance of correctly classifying new examples.

---
# Perceptron
The perceptron algorithm converges after a finite number of steps when the training examples are linearly separable. If the data isn’t separable, it won’t converge and isn’t guaranteed to find the separator with the smallest training error.

## Algorithm:
On input $S_n = \{ (x^{(i)}, y^{(i)} ) \}_{i=1}^{n}$

Initialize:
- $k=0$. This counts the number of guesses on $\bar{\theta}$
- $\bar{\theta}^{(0)} = \bar{0}$. Decides where the hyperplane is and which side is positive/negative.

while there exists a mis-classified point:
$for \space i=1,...,n:$
    if $y^{(i)}(\bar{\theta}^{(k)}\cdot \bar{x}^{(i)}) \leq 0$:
        $\bar{\theta}^{(k + 1)} = \bar{\theta}^{(k)} + y^{(i)}\bar{x}^{(i)}$
        k++

## Algorithm Explained:
While there is a point that is misclassified, we will enter the loop. We check if a point is mis-classified by seeing if $y^{(i)} (\bar{\theta} \cdot \bar{x}^{(i)}) \leq 0$.

We then enter a for loop that looks at each individual datapoint of our training data. If that datapoint is incorrectly predicted (i.e. $y^{(i)}(\bar{\theta}^{(k)}\cdot \bar{x}^{(i)}) \leq 0$), then we will enter the if statement. Otherwise, we do nothing.

If we enter into the if statement, that means that the datapoint was incorrectly classified. We need to fix this, so we update the value of $\bar{\theta}$. We update it with the equation $\bar{\theta}^{(k + 1)} = \bar{\theta}^{(k)} + y^{(i)}\bar{x}^{(i)}$.

There are two cases here.
- Case one: the datapoint was mislabeled as -1 when it should be +1. In this case, the actual label was +1, so $y^{(i)} = +1$. Therefore, we get $\bar{\theta}^{(k + 1)} = \bar{\theta}^{(k)} + (+1)\bar{x}^{(i)}$, which is the same as $\bar{\theta}^{(k + 1)} = \bar{\theta}^{(k)} + \bar{x}^{(i)}$. We just add the value of $\bar{x}^{(i)}$ to $\bar{\theta}$.
- Case two: the datapoint was mislabeled as +1 when it should be -1. In this case, the actual label was -1, so $y^{(i)} = -1$. Therefore, we get $\bar{\theta}^{(k + 1)} = \bar{\theta}^{(k)} + (-1)\bar{x}^{(i)}$, which is the same as $\bar{\theta}^{(k + 1)} = \bar{\theta}^{(k)} - \bar{x}^{(i)}$. We just subtract the value of $\bar{x}^{(i)}$ from $\bar{\theta}$.

## Why does the update $\bar{\theta}^{(k + 1)} = \bar{\theta}^{(k)} + y^{(i)}\bar{x}^{(i)}$ make sense?
It makes sense that we should want a point that was incorrectly classified to be more correctly classified after we update $\bar{\theta}$.

Let's look what happens after we update $\bar{\theta}$: $\bar{\theta}^{(k + 1)} = \bar{\theta}^{(k)} + y^{(i)}\bar{x}^{(i)}$

The next time we check if that point $x^{(i)}$ is mis-classified we look at: $y^{(i)}(\bar{\theta}^{(k + 1)}\cdot \bar{x}^{(i)}) \leq 0$. Note we are knowing using the (k+1) value of $\bar{\theta}$ instead of the kth. We found $\bar{\theta}^{(k+1)}$ with $\bar{\theta}^{(k + 1)} = \bar{\theta}^{(k)} + y^{(i)}\bar{x}^{(i)}$, so let's plug this in and expand.
$y^{(i)}(\bar{\theta}^{(k + 1)}\cdot \bar{x}^{(i)})$
$= y^{(i)}[(\bar{\theta}^{(k)} + y^{(i)}\bar{x}^{(i)})]\cdot \bar{x}^{(i)}$ by replacing $\bar{\theta}^{(k+1)}$
$= y^{(i)}[\bar{\theta}^{(k)}\cdot \bar{x}^{(i)} + y^{(i)}\bar{x}^{(i)}\cdot \bar{x}^{(i)}]$ by distributing over the dot product
$= y^{(i)}\bar{\theta}^{(k)}\cdot \bar{x}^{(i)} + y^{(i)^2}\bar{x}^{(i)}\cdot \bar{x}^{(i)}$ by distributing $y^{(i)}$

We can then break this down into two components:

**Part 1:** $y^{(i)}\bar{\theta}^{(k)}\cdot \bar{x}^{(i)}$
This is the original guess $\bar{\theta}^{(k)}$. It was $\leq 0$ since the data point was mis-predicted (as a reminder - we are only updating when we have a mis-predicted data point).

**Part 2:** $y^{(i)^2}\bar{x}^{(i)}\cdot \bar{x}^{(i)}$.
$y^{(i)^2}\bar{x}^{(i)}\cdot \bar{x}^{(i)}$ will always be a positive value (dot product of a vector with itself is positive)

**Putting it together**
Part 1 will be a negative value or 0. Part 2 will be a positive value. Therefore, when we add parts 1 and 2, we get either a positive value, or at least a less negative value. If we remember the original equation we were expanding, we can see that the left-hand side of $y^{(i)}(\bar{\theta}^{(k + 1)}\cdot \bar{x}^{(i)}) \leq 0$ will become more positive after the update, therefore meaning the point becomes better classified.

---

# Perceptron with Offset
**Goal:** learn decision boundary $h(\bar{x};\bar{\theta}) = \text{sign}(\bar{\theta}\cdot\bar{x} + b)$ that minimizes training error:
$E_n(\bar{\theta}) = \frac{1}{n} \sum_{i=1}^{n} [[ y^{(i)}(\bar{\theta}^{(k)}\cdot \bar{x}^{(i)} + b) \leq 0]]$
We can now fully define a line:

![[linear-classifier---math-20220319180958559.png]]
In order to account for the additional offset term, we can augment the $\bar{\theta}$ and the feature vectors $\bar{x}$ to have an additional dimension:
$\bar{\theta} = [b, \theta_1, \theta_2, ... \theta_d]$
$\bar{x} = [1, x_1^{(1)}, ..., x_d^{(1)}]$
This will update the offset term as $b^{(k + 1)} = b^{(k)} + y$ each update.