---
tags: [flashcards]
source:
summary: the softmax function is often used as the last activation when dealing with multi-class prediction. The outputs will be normalized to `[0, 1]` and the total probabilities will sum to 1.
---

> [!NOTE]
> There isn't a visualization of the softmax function because it operates on multiple variables so it is hard to visualize. See [[Sigmoid]] for a visualization of what softmax looks like for binary classification (ex. hot dog or not) since the two are equivalent for binary classification.


> [!NOTE] When to use softmax vs. sigmoid?
> Use softmax for multiple-class prediction and sigmoid for binary prediction. See the bottom of the page for more details.

### Softmax Function
The softmax function is used to convert the raw scores given by the classifier to **logits**, or normalized scores that add up to 1. Softmax has the nice property that it can take in negative values and ensure the output is positive and that all the outputs add up to 1 (converts everything to probabilities).

It is called softmax because it is like a softer version of a max function. It will give a higher probability to the larger value.
$$
\hat{y}_c = \frac{e^{z_c}}{\sum_j e^{z_j}}
$$

**Calculation:**
1. First you take the exponential of the raw scores $z_c$ with $e^{z_c}$. This ensures all values are positive and non-zero. The new values are $e^{z_j}$.
2. You then sum up all the positive scores $\sum_j e^{z_j}$ and divide the positive scores calculated in step 1, $e^{z_j}$, by the sum of the scores, to normalize them to add up to 1.

**PyTorch Example**:
```python
m = nn.Softmax(dim=1)
input = torch.randn(2, 3)
output = m(input)
```

### When to use [[Sigmoid]] vs. softmax
If you have a multi-label classification problem = there is more than one "right answer" = the outputs are NOT mutually exclusive, then use a sigmoid function on each raw output independently. The sigmoid will allow you to have high probability for all of your classes, some of them, or none of them. Example: classifying diseases in a chest x-ray image. The image might contain pneumonia, emphysema, and/or cancer, or none of those findings.

If you have a multi-class classification problem = there is only one "right answer" = the outputs are mutually exclusive, then use a softmax function. The softmax will enforce that the sum of the probabilities of your output classes are equal to one, so in order to increase the probability of a particular class, your model must correspondingly decrease the probability of at least one of the other classes. Example: classifying images from the MNIST data set of handwritten digits. A single picture of a digit has only one true identity - the picture cannot be a 7 and an 8 at the same time.

**For binary classification, sigmoid and softmax are the same:**
Using [[Sigmoid]] with dummy encoded output (one binary column) vs using softmax with two one-hot encoded columns (one of the columns is equal to one, the other is zero) is [mathematically equivalent](https://stats.stackexchange.com/q/233658/35989) and should give same results. [Source](https://stats.stackexchange.com/a/520856/271266).