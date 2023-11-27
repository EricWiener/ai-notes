---
tags: [flashcards]
source: [[Deep Fried Convnets.pdf]]
summary:
---


# Deep Fried Convnets

PDF: https://drive.google.com/open?id=1F08y7NBPJUS-Uo6rW8jPw1MyC3ayS1pE&authuser=ecwiener%40umich.edu&usp=drive_fs
Reviewed: No
url: https://arxiv.org/abs/1412.7149

### Abstract + Intro

- The fully-connected layers of deep convolutional neural networks typically contain over 90% of the network parameters.
- Adaptive Fastfood transform to reparameterize the matrix-vector multiplication of fully connected layers
- Convolutional layers, which contain a small fraction of the network parameters, represent most of the computational effort.
- Fully connected layers contain the vast majority of the parameters but are comparatively cheap to evaluate.
- Existing approaches realize speed gains at test time but do not address the issue of training, since the approximations are made after the network has been fully trained. Additionally, existing approaches focus on convolutional layers for speed increases, but don’t address the total number of parameters very well since most of these are in the FC layers.
- There is significant redundancy in the parameterization of several deep learning models.

> [!note]
> In this paper we show how the number of parameters required to represent a deep convolutional neural network can be substantially reduced without sacrificing predictive
> 

- Convolutional layers are much more expensive to evaluate than fully connected layers, so replacing fully connected layers with more convolutions can decrease model size but comes at the cost of increased evaluation time.

### Related Work

- Network in Network architecture of [[25] achieves state of the art results on several deep learning benchmarks by replacing the fully connected layers with global average pooling. [Notion link]]

### Question: how to use a pre-trained model when using global average pooling?

Hi, does anyone know the best practice for taking a pre-trained network with global average pooling as the final layer and then re-using it for another task with a different number of classes (if you are just using global average pooling as the final layer with no FC as in [Network in Network](https://arxiv.org/abs/1312.4400)). In Network in Network, they say:

> The idea is to generate one feature map for each corresponding category of the classification task in the last mlpconv layer. Instead of adding fully connected layers on top of the feature maps, we take the average of each feature map, and the resulting vector is fed directly into the softmax layer.
> 

So it seems like you would just change the number of output channels for the global average pooling (and the conv layer before it). However, [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) says:

> The use of average pooling before the classifier is based on [12], although our implementation differs in that we use an extra linear layer. This enables adapting and fine-tuning our networks for other label sets easily, but it is mostly convenience and we do not expect it to have a major effect.
> 

And then [Deep Fried Convnets](https://arxiv.org/abs/1412.7149) references Going Deeper with Convolutions:

> It is very common in practice to take a convolutional network trained on ImageNet and re-train the top layer on a different data set, re-using the features learned from ImageNet for the new task (potentially with fine-tuning), and this is difficult with global average pooling. This deficiency is noted by [35], and motivates them to add an extra linear layer to the top of their network to enable them to more easily adapt and fine tune their network to other label sets.
> 

Deep Fried Convnets makes it seem like it's quite challenging to re-use features from another model if global average pooling is used, but the reference they use (to Going Deeper with Convolutions) makes it seem as if they just used a FC layer as a convenience and didn't make it seem like a difficulty. Is there any reason why adding a FC layer is easier than re-training the last conv layer before the global average pooling layer?