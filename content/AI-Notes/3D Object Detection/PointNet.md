---
tags: [flashcards]
source: https://towardsdatascience.com/deep-learning-on-point-clouds-implementing-pointnet-in-google-colab-1fd65cd3a263
summary: a neural network that uses point cloud data to solve object classification, part segmentation, and semantic segmentation.
---

[Arvix](https://arxiv.org/abs/1612.00593)

PointNet is a neural network that uses point cloud data to solve object classification, part segmentation, and semantic segmentation.

![[AI-Notes/3D Object Detection/PointNet/Untitled.png]]

- **Classification:** given a point cloud for an object, what type of object is it?
- **Part Segmentation:** given a point cloud for an object, what are the different parts in it?
- **Semantic Segmentation:** given a point cloud of multiple objects, which object is which?

In order for any network to work with point clouds, it must meet the following constraints:

- **Invariant to permutations:** because point clouds are unordered (you just get a random set of points), the algorithm should give the same results no matter what order the points come in.
- **Invariant to rigid transformations:** if you rotate or translate (move) an object, it should still be classified as the same object
- The network should capture interaction between points

## Architecture (simplified view)

![[AI-Notes/3D Object Detection/PointNet/Screen_Shot_1.png]]

1. You pass the points through an MLP (multi-layer perceptron / fully connected network) and generate feature vectors for each point.
2. You then perform max-pooling over all these feature vectors to get a pooled feature vector
3. You then pass the pooled feature vector through a fully connected layer to get class scores