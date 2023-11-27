# Boosting, AdaBoost

Files: Lecture_12_-_Boosting.pdf
summary: Boosting will train models sequentially and try to improve on previous models by weighting points that previous models failed to classify. Decreases bias by improving models sequentially + decreases variance by combining final models.

[Helpful Kaggle Article](https://www.kaggle.com/prashant111/bagging-vs-boosting)

[Another helpful article with good diagrams](https://quantdare.com/what-is-the-difference-between-bagging-and-boosting/)

- Boosting is a general strategy for combining weak classifiers into a strong classifier. A weak classifier has a classification rate > 50%
- You assign weights to both the individual samples and individual classifiers.
- Output decision is weighted decision of individual weak classifier outputs
- Assigns a high weight to hard to classify points and a low weight to easy to classify points
- Boosting is a sequential process, where **each subsequent model attempts to correct the errors of the previous model**. The succeeding models are dependent on the previous model.
- In this technique, learners are **learned sequentially** with early learners fitting simple models to the data and then analyzing data for errors. In other words, we fit consecutive trees (random sample) and at every step, the goal is to solve for net error from the prior tree.
- When an input is misclassified by a hypothesis, its weight is increased so that next hypothesis is more likely to classify it correctly. By combining the whole set at the end converts weak learners into better performing model.

## Bagging vs. Boosting

> [!note]
> Bagging will train models in parallel and then combine all models for a final vote. Boosting will train models sequentially and try to improve on previous models by weighting points that previous models failed to classify.
> 

![[/Screenshot 2022-02-04 at 09.31.32@2x.png]]

> [!note]
> Note that with boosting you can either train on the entire dataset each time or you can use your weighted values to create random subsamples with weighted probability.
> 

## **Boosting Example (from Kaggle)**

- Let’s understand the way boosting works in the below steps.
    1. A subset is created from the original dataset.
    2. Initially, all data points are given equal weights.
    3. A base model is created on this subset.
    4. This model is used to make predictions on the whole dataset.

![https://miro.medium.com/max/171/0*u3Li30F4gRAV_3Fb.png](https://miro.medium.com/max/171/0*u3Li30F4gRAV_3Fb.png)

1. Errors are calculated using the actual values and predicted values.
2. The observations which are incorrectly predicted, are given higher weights. (Here, the three misclassified blue-plus points will be given higher weights)
3. Another model is created and predictions are made on the dataset. (This model tries to correct the errors from the previous model)

![https://miro.medium.com/max/166/0*yRk4nLMrvoA4cvC6.png](https://miro.medium.com/max/166/0*yRk4nLMrvoA4cvC6.png)

1. Similarly, multiple models are created, each correcting the errors of the previous model.
2. The final model (strong learner) is the weighted mean of all the models (weak learners).

![https://miro.medium.com/max/1202/1*k-HYpwcgzCq_Yy--05_LAw.png](https://miro.medium.com/max/1202/1*k-HYpwcgzCq_Yy--05_LAw.png)

- Thus, the boosting algorithm combines a number of weak learners to form a strong learner.
- The individual models would not perform well on the entire dataset, but they work well for some part of the dataset.
- Thus, each model actually boosts the performance of the ensemble.

![https://miro.medium.com/max/180/0*AHlYVBCC5mpDCedP.png](https://miro.medium.com/max/180/0*AHlYVBCC5mpDCedP.png)

## AdaptiveBoosting (AdaBoost)

> [!note]
> This is a famous model that uses boosting and gets good results.
> 

- **Is the entire dataset used to train each new model or a subset?**
    
    There are two methods for training Adaboost. Either use the weight vector directly in the training of the weak learner, or use the weight vector to sample datapoints with replacement from the original data. [Source](https://stats.stackexchange.com/questions/45233/regarding-the-sampling-procedure-in-adaboost-algorithm)
    
- Schapire initially answered whether you can combine a bunch of weak classifiers to make a better classifier
- AdaBoost is one of the best out-of-the-box classifiers. Developed by Fruend and Schapire and won 2003 Gödel Prize.
- Assigns a high weight to hard to classify points and a low weight to easy to classify points

**Algorithm:** *Initialize*:

- Given training data examples $\{(x^{(i)}, y^{(i)}\}^n_{i=1}$
- Assign uniform weights $w_0(i)$ to data $i = 1, ..., n$

$h(\bar{x}; \bar{\theta}_m) = sign(\bar{\theta_1}(x_k - \theta_0))$ where $\bar{\theta} = (k, \theta_0, \theta_1)$ - (feat #, pos, direction)
Note that $\bar{\theta} \in \mathbb{R}^3$ where $\bar{\theta} \in \{1, ..., d\} \times \mathbb{R} \times \{-1, +1\}$

![[AI-Notes/Video/Untitled]]

![[AI-Notes/Video/Untitled]]

*Output classifier:*
sign($h_M(\bar{x})$) where $h_M(\bar{x}) = \sum_{m = 1}^M \alpha_m h(\bar{x}; \bar{\theta}_m)$. The output classifier is the weighted average of the **outputs** of the individual weak classifiers.

**Notes:**

- When assigning a weight to a classifier with $\alpha_m = \frac{1}{2}\ln(\frac{1 - \hat{\mathcal{E}}_m}{\hat{\mathcal{E}_m}})$
    - If you have exactly 50% accuracy, then you have ln(.5/.5) = ln(1) = 0. This classifier gets no weight.
    - If you have perfect accuracy, then you have ln(1/0) → infinity.
    - We don’t allow negative $\alpha_m$. We don’t allow there to be more incorrectly classified than correctly classified because this would force $\alpha_m$ to be negative.
- The updated weights uses the exponential loss. We don’t need to recalculate the weights each time, but can just update it by the current weights (no need to look at the previous classifiers again), but this exponential loss is still in regards to the ensemble classifier. This is because $e^{2 + 3} = e^2 * e^3$, so you can just calculate the new weights and multiply it by the previous weights and this will be the new exponential loss.
- Guaranteed that no consecutive decision stumps are the same - this is assuming 50% < accuracy < 100%.
- AdaBoost can’t perfectly classify XOR using decision tree stumps with max depth 1. However, AdaBoost can be used with any weak classifiers.

We update the weights with $exp(-y^{(i)}\alpha_mh(\bar{x}^{(i)}; \bar{\theta}_m))\bar{w}_{m-1}(i)$

- The negative is because the graph of $e^{-x}$ looks like the graph on the right. When the prediction is negative, the loss will be greater. When the prediction is positive, the loss will approach zero.

![[AI-Notes/Video/Untitled]]

![[AI-Notes/Video/Untitled]]