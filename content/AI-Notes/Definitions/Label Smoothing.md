---
tags: [flashcards]
source: https://paperswithcode.com/method/label-smoothing
summary: replace a one-hot encoding with values slightly greater than 0 and slightly less than 1.
---

Label smoothing replaces the ==0 and 1== classification targets of a one-hot encoding with targets of $\frac{\epsilon}{k-1}$ and $1-\boldsymbol{\epsilon}$ respectively to add ==noise to labels== and avoid overfitting/account for incorrect labels.
<!--SR:!2028-05-06,1756,350!2025-10-11,935,330-->

**Label Smoothing** is a regularization technique that introduces noise for the labels. This accounts for the fact that datasets may have mistakes in them, so maximizing the likelihood of $\log p(y \mid x)$ directly can be harmful. Assume for a small constant $\boldsymbol{\epsilon}$, the training set label y is correct with probability $1-\boldsymbol{\epsilon}$ and incorrect otherwise. Label Smoothing regularizes a model based on a [[Cross Entropy Loss#Soft-Max Function|softmax]] with $k$ output values by replacing the hard 0 and 1 classification targets with targets of $\frac{\epsilon}{k-1}$ and $1-\boldsymbol{\epsilon}$ respectively.