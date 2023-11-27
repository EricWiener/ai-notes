# Bagging, Random Forests

Files: Lecture_11_-_Bagging.pdf
summary: Bagging will train models in parallel and then combine all models for a final vote. Decreases variance by combining final models.

![[AI-Notes/Video/Untitled]]

Note that the classifiers don’t necessarily need to be decision trees. You can use any type of classifier.

Number of bagged classifiers → ∞, misclassification rate → 0 **assuming**:

- Each decision tree has a classification rate greater than chance (> 0.5 for binary, > 0.33 for tri-nary, etc.)
- Classifiers are independent
    - This is difficult because you have trained all the classifiers on the same dataset.

**Issues:**

- Prediction error rarely goes to zero
- Bagging reduces variation (estimation error), but bias (structural error) remains.
- Independence of classifiers is a strong assumption

### Random Forests

Random forests are just like bagging, except we randomly select *k* features from *d* features. Another option is to select the *k* features from the top *m* features (a subset of the original *d* features).

Select the best features to split on using information gain.

Enforce independence assumption of the bagged classifier.