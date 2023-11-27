---
tags: [flashcards, eecs498-dl4cv]
source:
summary: kNN classifier predicts labels based on nearest training examples (using k nearest neighbors).
---
The most simple classifier is the nearest neighbor classifier. You just predict the label of the most similar image. You just memorize all the data and the labels.

We need a distance metric to compare images. We can use **L1 distance** (manhattan distance): sums up the magnitude of the difference of all pixels

![[AI-Notes/Concepts/k-nearest-neighbor-srcs/Screen_Shot.png]]

![[AI-Notes/Concepts/k-nearest-neighbor-srcs/Screen_Shot 1.png]]

For the nearest neighbor classifier, training is $O(1)$ and is very fast. Prediction is $O(N)$. This is **very bad**. We can afford slow training, but we need fast inference. Note that the `predict` step loops over all `n` test input images (this is why the loop is needed and why `Ypred` is a vector)

# K-Nearest Neighbors

> [!note]
> kNN classifier predicts labels based on nearest training examples (using **k** nearest neighbors).
> 
![[AI-Notes/Concepts/k-nearest-neighbor-srcs/Screen_Shot 2.png]]
In the above graph the color indicates the label of a point. The points are training examples that consist of an (x, y) coordinate. The background colors give the category a test point would be assigned if it were picked in that region. When $K > 1$, you can have ties between classes (the white regions) and you need to break ties.

The more neighbors, the smoother the boundaries (and you don’t end up with outliers causing random islands of a different label like the yellow in the green on the left). We can change the behavior by adjusting how many neighbors we want to consider.

### **L1 vs L2 Distance:**
[[Regularization]]

We can also change the distance metric. 

![[l1-and-l2-distance-from-origin.png]]
This graph is showing the points on the plane that have a distance of 1 from the origin as computed with the respective distance function.

**L1** takes the sum of the absolute value of the differences. **L2** takes the square root of the sum of the squared difference of the errors. **Note:** L2 does sqrt(sum(...)). If it did sum(sqrt(...)) then it would be equivalent to L1).

Note that the L1 distance looks like a rotated square because you add up that $\Delta x$ and $\Delta y$ to calculate the distance, while L2 calculates the straight line distance between points.

![[AI-Notes/Concepts/k-nearest-neighbor-srcs/Screen_Shot_2.png]]

L2 is a coordinate frame free metric. If we rotate the coordinate frame, the behavior will remain the same (rotating a circle doesn’t change anything). Note that in the L2 colored graph, the decision boundaries can appear at any angle.

L1 is used when there is something special about the particular axises of your data. If you rotated the axises a bit, the classification would change. For L1, decision boundaries are either aligned to the axises of 45 degrees to it.

If we take our dataset and perform a 15 degree rotation, the L2 boundaries would be rotated accordingly. For L1, the new boundaries would be totally different.

> [!note]
> L1 is more robust to outliers and has decision boundaries aligned to or $45  \degree$ to coordinate frame. L2 can have decision boundaries at any angle and doesn’t change when the dataset is transformed.
> 

### kNN: Distance Metric

With the right choice of distance metric, you can apply k-Nearest Neighbor to any type of data.

- Ex: you can compare papers using [[tf-idf similarity.]]

### kNN: Usage

The best value of **K** and the best distance measure to use are **hyperparameters.** They are choices about our algorithm that aren't learned directly from the training set. 

**K-Nearest Neighbors is rarely used on images.** It's very easy to distort the image slightly and get a totally different label. It's also very slow at test time. It also takes up a lot of memory to memorize all the training data.

![[AI-Notes/Concepts/k-nearest-neighbor-srcs/Screen_Shot 3.png]]

However, kNN with features (instead of raw pixels) can work well. For instance, you can perform nearest neighbors on ConvNet features. Although it can be hard to find a good similarity metric for images. Additionally, it will give equal weight to all features equally.

**Universal approximation**: as the **number of training examples** goes to infinity, nearest neighbor can represent any function (subject to some conditions)

**Curse of dimensionality:** as you increase the number of spacial dimensions, the number of points needed to maintain the same spacing needs to increase exponentially.

[[K-Nearest Neighbors]]

![[curse-of-dimensionality.png]]
Note this is referring to k-NN and maintaining the same spacing between data points when trying to find the most similar points. This is also applicable to other similar algorithms.

> [!note]
> The curse of dimensionality makes it so the amount of data we would need to achieve universal approximation for a high dimensional dataset is way too much to be feasible.
> 