---
tags:
  - flashcards
summary: Replaces FC layer with global average pooling (introduces global average pooling)
source: https://arxiv.org/abs/1312.4400
---
Network in Network refers to the paper using three mlplayers and within each mlpconv layer, there is a three-layer perceptron.

# Global Average Pooling
- Fully connected layers are prone to overfitting, thus hampering the generalization ability of the overall network. Dropout can be used to improve generalization ability and reduce overfitting.
- This paper proposes global average pooling to replace the traditional fully connected layers at the ends of CNNs.
- The idea is to generate one feature map for each corresponding category of the classification task in the last mlpconv layer. Instead of adding fully connected layers on top of the feature maps, we take the average of each feature map, and the resulting vector is fed directly into the softmax layer.
- More native to the convolution structure by enforcing correspondences between feature maps and categories. The feature maps can be easily interpreted as categories confidence maps.

# Experiments
- Use global average pooling instead of fully connected layers at the top of the network

### Global average pooling as a regularizer
- We replace the global average pooling layer with a fully connected layer, while the other parts of the model remain the same.
- An equivalent network with global average pooling is then created by replacing the dropout + fully connected layer with global average pooling

# Questions:
- [ ]  What is meant by “model discriminability for local patches within the receptive field”. [SO question](https://ai.stackexchange.com/questions/8490/what-is-meant-by-model-discriminability-for-local-patches-within-the-receptive)