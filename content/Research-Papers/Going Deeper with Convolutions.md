---
tags: [flashcards]
source: [[Going Deeper with Convolutions.pdf]]
aliases: [GoogleNet, Inception Networks]
summary: propose a deep convolutional neural network architecture codenamed Inception, which was responsible for setting the new state of the art for classification and detection. One particular submission they used for the ImageNet competition was called GoogleNet (won in 2014).
---
- [https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202)
- The use of average pooling before the classifier is based on [12], although our implementation differs in that we use an extra linear layer. This enables adapting and fine-tuning our networks for other label sets easily, but it is mostly convenience and we do not expect it to have a major effect.

**From ReNeXt:**
![[ResNeXt#Reference to Going Deeper with Convolutions GoogleNet]]

### Notes from Deepfried Convnets
- It is very common in practice to take a convolutional network trained on ImageNet and re-train the top layer on a different data set, re-using the features learned from ImageNet for the new task (potentially with fine-tuning), and this is difficult with global average pooling. This deficiency is noted by [35], and motivates them to add an extra linear layer to the top of their network to enable them to more easily adapt and fine tune their network to other label sets

# GoogLeNet: 2014
This won the 2014 challenge (same year as VGG). This was focused on making the network more efficient. Google has a ton of data but also runs in limited environments (ex. on phones). They wanted to reduce the size of the network as much as possible while maintaining high accuracies.

**Stem network:** at the start aggressively downsample the input. The very expensive layers are the ones at the start, so they quickly downsample this. The stem downsampled from 224 to 28 in just a few layers. 
![[googlenet-stem-network.png]]
At the first layer of the network it was cheaper to use a single convolutional layer with a larger kernel vs. multiple 3x3 convolutional layers. This is because the input to the network always has 3 channels (RGB) while latter layers will usually have more channels. Specifically, here we had an input of $224 \times 224 \times 3$ and GoogleNet used a kernel size of 7 (stride 2 + pad 3). FLOPs is calculated with $C_{\text{out}} \times C_{\text{in}} \times K_H \times K_W \times \mathrm{H}_{\text{out}} \times \mathrm{W}_{\text{out}}$ so **for a small number of input channels and large height and width, it is better to shrink height and width than shrink channels.**

**Inception module:** local structure that was repeated throughout the entire network. Just as VGG got rid of the kernel size being a parameter by only using 3x3, Google also got rid of the kernel size as a parameter. However, they did this by using every kernel size. Within a single inception module, it does a 1x1 convolution, 3x3 convolution, 5x5 convolution, and a max pooling.

![[AI-Notes/Concepts/cnn-architectures-srcs/Screen_Shot_1.png|500]] ![[AI-Notes/Concepts/cnn-architectures-srcs/Screen_Shot 3.png|500]]


They also used 1x1 convolutions in the inception module to reduce the number of channels before an expensive convolution (convolution with a filter with a larger kernel size). These 1x1 layers are referred to as “Bottleneck” layers.

> [!note]
> Previous models all had linear structures where one convolutional layer followed another one. GoogLeNet introduced the idea of having multiple branches.
> 

**Global Average Pooling:**

The majority of parameters in AlexNet and VGG came from the large fully connected layers at the end of the model. Google got rid of these and instead used **global average pooling** to collapse spatial dimension (vs. AlexNet and VGG which just flattened the spatial dimensions), and one linear layer to produce class scores. Note that the pooling is still being applied independently to each channel.

![[Screenshot_2022-02-01_at_08.22.552x.png]]

They get rid of the 7x7 spatial dimensions by using average pooling with a kernel of size 7x7 to transform the 7x7x1024 input to 1x1x1024.

- The output of the last CONV layer before the global average pooling was $7 \times 7$ with 1024 channels.
- They then apply a global average pooling with filter size $7 \times 7$ over all of the 1024 channels, which results in 1024 values.
- They then feed these 1024 values to a fully connected layer which gives the score for 1000 classes.

> [!note]
> GoogLeNet got rid of the large fully connected layers at the end (which have a lot of parameters). Most modern networks use gobal average pooling.
> 

**Auxiliary Classifiers**

This was at a time before batch normalization, so it was very difficult to train deep networks because the gradients don't propagate cleanly (due to vanishing and exploding gradients) . As a hack, they attached "auxiliary classifiers" at several intermediate points in the network that also try to classify the image and receive loss.

Auxiliary classifiers are not used anymore.

**Batch norm was discovered between 2014 and 2015.**

![[AI-Notes/Concepts/cnn-architectures-srcs/Screen_Shot_2.png]]

> [!note]
> Unless your values in your matrices are all exactly 1s, you will either make your gradient smaller each time you multiply it or larger. This leads to vanishing/exploding gradients very quickly.
> 
