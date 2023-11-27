---
tags: [flashcards, eecs498-dl4cv]
source:
summary: Choices about our algorithm that aren't learned directly from the training set.
---

**Hyperparameters** are properties of the model. For a kNN classifier, they would include the choice of $k$, the distance function used, etc. **They are choices about our algorithm that aren't learned directly from the training set.** 

**Idea 1:** To choose the best hyperparameters, you can't just choose the ones that work best on the data. If you choose **k = 1**, it will perform perfectly on the training data. (Don't do).

**Idea 2:** We need to split the data into **train** and **test.** We then choose the hyperparameters that work best on test data. This doesn't work well because if you test out too many hyperparameters on the test set, then you are training on the test set.

### **Idea 3: We check the hyperparameters with the validation set.**

We then evaluate the final choice on the test set. We should only evaluate on the test set once. 

![[train-val-test-splits.png]]

> [!note]
> You don’t want to make **hyperparameter decisions** based on performance on the test set or you will effectively be training on the test set (trying out multiple models and choosing the one that performs the best on the test data). It is better to evaluate on validation data and then only evaluate on the test data once.
### **Idea 4:** Cross-validation.

Split the data into **folds.** Train on subsets of folds and use the other for validation. You can then average the results. This is the best way, but it very computationally expensive because you need to train on each subset of folds. 

![[cross-validation-diagram.png]]

- Note that you throw out your model weights after each split. This is used to tell you how good a model architecture is - not how good a certain trained model is.

> [!note]
> Note that this tells you how good your hyperparameters are and how well your model generalizes, but doesn't help optimize the model. Only the loss is used during training time to optimize the model. **The validation data is not used to optimize the model.**
> 
![[cross-validation-for-k.png]]

Example of running 5-fold cross-validation to find the best value of $k$ (a hyperparameter). It seems that a k of 7 works best since it seems the trials with k = 7 has best overall performance. The variance in the performance is due to different training and validation data.