---
tags: [flashcards]
source:
summary:
---

> [!note]
> An SVM with a non-linear kernel is basically a linear classifier, but it just transforms the data.
> 

### [[Collaborative Filtering - kNN]] vs. SVM for embedding lookups
[Tweet](https://twitter.com/karpathy/status/1647025230546886658?lang=en) from Andrej Karpathy on k-Nearest Neighbor lookups on embeddings: in my experience much better results can be obtained by training SVMs instead. Not too widely known. Works because SVM ranking considers the unique aspects of your query w.r.t. data.

[Example Notebook](https://github.com/karpathy/randomfun/blob/master/knn_vs_svm.ipynb) where he trains an SVM with the query as the one positive class:
```python
from sklearn import svm

# create the "Dataset"
x = np.concatenate([query[None,...], embeddings]) # x is (1001, 1536) array, with query now as the first row
y = np.zeros(1001)
y[0] = 1 # we have a single positive example, mark it as such

# train our (Exemplar) SVM
# docs: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=0.1)
clf.fit(x, y) # train
```

**Why does this work?** In simple terms, because SVM considers the entire cloud of data as it optimizes for the hyperplane that "pulls apart" your positives (query) from negatives (embeddings to query over). In comparison, the kNN approach doesn't consider the global manifold structure of your entire dataset and "values" every dimension equally. The SVM basically finds the way that your positive example is unique in the dataset, and then only considers its unique qualities when ranking all the other examples.

### ChatGPT on SVM vs [[Collaborative Filtering - kNN]]
When comparing the performances of Support Vector Machines (SVMs) and k-Nearest Neighbors (kNN) for query embedding lookup, it's important to understand the underlying principles and strengths of each method.

1. **High-dimensional Spaces:** In many real-world applications, the feature vectors (like query embeddings in NLP) are high-dimensional. kNN often suffers from the "curse of dimensionality" where the distance between nearest and farthest neighbors becomes indistinguishable as the dimensionality increases. SVM, on the other hand, uses hyperplanes and is thus more effective in high-dimensional spaces.
    
2. **Training and Prediction Time:** kNN is a lazy learning method, which means it does not "learn" anything from the training data and simply stores it. Thus, while training time is minimal, prediction time can be very high as the model has to compute distances to all points in the training dataset to make a prediction. SVM, in contrast, only requires the support vectors to make a prediction, making it faster for large datasets.
    
3. **Boundary Decision:** SVMs are known for their ability to construct an optimal hyperplane that maximizes the margin between different classes, which can be particularly useful if the classes are linearly separable or close to it. On the other hand, kNN's decision boundary can be more irregular and sensitive to noisy instances.
    
4. **Overfitting:** SVMs are less prone to overfitting because they focus on the points that are difficult to classify, ignoring those that are easy to classify. kNN, however, can overfit to noise in the data, especially with a small value of k.
    
5. **Non-linear Separation:** SVMs can also handle non-linear separation by projecting the data into a higher-dimensional space using the "kernel trick". While kNN can also handle non-linear boundaries, this comes at the cost of complex computation and irregular boundaries.
