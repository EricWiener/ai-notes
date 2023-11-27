---
summary: History of CNN Architectures - AlexNet, VGG, GoogLeNet, ResNet
tags:
  - EECS-498-F20
---

# AlexNet
> [!note]
> When people refer to using an AlexNet architecture, they usually refer to a re-implementation that came out pretty soon after the paper was released and not to the model in the original paper (since the model in the original paper was a little fishy with incorrect dimensions).
> 

ImageNet classification scores are shown to the right. In 2010 and 2011, the winning models used hand-crafted features with a linear classifier. AlexNet was the first model to use a Convolutional Neural Network and it performed extremely well. Sparked a lot of interest in the field (more citations than Darwin's "On the Origin of Species").
![[AI-Notes/Concepts/cnn-architectures-srcs/Screen_Shot.png]]
Error rate on ImageNet challenge

**Architecture:**

- 227 x 227 inputs
- 5 convolutional layers
- Max pooling
- 3 fully-connected layers
- ReLU nonlinearities (between convolutional layers)

**Interesting Info**

- Needed to split the model over 2 GPUs because at the time GPUs only had about 3 GB of memory (now Colab GPUs have approximately 12-16 GB).
- Used "Local Response Normalization" which was a precursor to BatchNorm. Not really used anymore.
- The diagram used in the paper got cut off at the top (kinda funny such an influential paper has a messed up diagram)
- The input dimensions shown in the paper are 224x224, but these don’t match future layers. The 227x227 dimensions described above come from a famous re-implementation.

### Architecture

![[AI-Notes/Concepts/cnn-architectures-srcs/Screen_Shot 1.png]]

The architecture was decided on with a lot of trial and error. However, we can look at trends in amount of memory (of the **output of that layer - not of the weights**), the number of params (learnable weights), the number of floating point operations (flops).

![[AI-Notes/Concepts/cnn-architectures-srcs/Screen_Shot 2.png]]

- All of the pooling layers take so few floating operations that they effectively round down to zero
- Most of the memory usage is in the **early convolutional layer outputs.**  (the earlier outputs have a relatively high resolution and high number of filters/channels).
- Nearly all parameters are in the **fully-connected layers**. The majority are in the first fully connected layer.
- Most floating-point operations occur in the convolutional layers. The fully connected layers are just multiplying a large matrix, while the CNN layers are computing many dot products. There are more flops if the CNN output has a high resolution and many layers.

**Caffe**
- After AlexNet came out, the original source code was gross and a custom thing that Alex built.
- People at UC Berkley were excited by the results and built a clean library called Caffe for machine leaning. This was used for a long time. It isn’t widely used anymore.

# ZFNet: A bigger AlexNet (2013)
This was the winning submission in 2013 for ImageNet. This year, almost all models were Neural Network based. ZFNet is a bigger AlexNet based on more trial and less error. Just tweaked some settings.

It does less aggressive downsampling than AlexNet, so all the remaining layers will have a higher spatial resolution. This allows more parameters and takes more memory. Bigger networks tend to work better.

# VGG: Deeper Networks, Regular Design (2014)
This was one of the first architectures that had a **principled design** throughout. AlexNet and ZFNet were made through trial and error, which makes it hard to scale them up or down. **VGG** was dramatically bigger than AlexNet. It was made by a single grad student and advisor.

- All CONV are 3x3; stride 1; pad 1
- All max pools are 2x2; stride 2
- After a max pooling layer, double the number of channels.
- Has convolutional stages (series of convolutions and then pooling)

**All CONV are 3x3; stride 1; pad 1**

This was done because if you have a 5x5 convolution, it will have $25 C^2$ parameters and $25C^2HW$ floating point operations. Using two 3x3 convolutions, you will still end up with the **same** **receptive field**, but will only have $(2 * 3^2)C^2=18C^2$ parameters and $(2 * 3^2)C^2HW=18C^2HW$ floating point operations. This has fewer learnable parameters and fewer floating point operators. **You can stop considering the kernel size as a hyperparameter and instead focus on how many 3x3 filters you should have.** You can also add ReLU filters between the 3x3 filters, which provides more nonlinear computation.

- Note that $25C^2$ parameters comes from the weight matrix having dimension $C_{\text{out}} \times C_{\text{in}} \times K_H \times K_W$ but $C_{\text{out}} = C_{\text{in}}$ for VGG.
- You then have $25C^2HW$ floating point operations because you perform an inner product for each element in the output layer with height $H$ and width $W$(note that the channel dimension of the output layer is already included in the $C^2$ term).

Note that you although a 3x3 Gaussian kernel can be represented by two 1x3 kernels convolved with each other, you can't use two 1x3 filters to detect diagonal edges, while a 3x3 filter can. You can't shrink the filter down much more from a 3x3 and still detect the same things.

**All max pools are 2x2; stride 2 + after a max pooling layer, double the number of channels.**

This means every pooling layer will halve the spatial resolution of the input and twice as many channels. We want each convolutional layer to cost the same amount of floating point operations. We can do this by halving the spatial size and doubling the number of channels.

The authors wanted to make VGG deeper and have more learnable parameters. However, they also wanted to reduce the number of FLOPs (floating point operations) to speed up training. Consider passing an arbitrary input to a 3x3 convolutional layer where $C_{\text{out}} = C_{\text{in}}$ which we will denote as Conv(3x3, C→C):

- If the input has dimensions $\mathrm{C} \times 2 \mathrm{H} \times 2 \mathrm{W}$ (height and width are doubled to increase params), the stats of passing it through a Conv(3x3, C→C) layer will be:
    - Memory:$\text { 4HWC }$
    - Params: $9\mathrm{C}^2$
    - FLOPs: $36 \mathrm{HWC}^{2}$
- If the input has dimensions $2 \mathrm{C} \times \mathrm{H} \times \mathrm{W}$ (channel dimension is doubled), the stats of passing it through a Conv(3x3, C→C) layer will be:
    - Memory: $\text { 2HWC }$
    - Params: $36\mathrm{C}^2$
    - FLOPs: $36 \mathrm{HWC}^{2}$

> [!note]
> By doubling channels every time you halve the spatial dimensions, the convolutional layers have the same computational cost (in terms of FLOPs).
> 

**Training without batch norm**

Because batch norm wasn't a thing yet, VGG had to train a shallower version of VGG first and then insert additional layers between the already trained layers.

![[Going Deeper with Convolutions#GoogLeNet 2014]]

![[ResNet]]

# Comparing Complexity

![[AI-Notes/Concepts/cnn-architectures-srcs/Screen_Shot 9.png]]

- The size of the dot is the number of learnable parameters
- The y-axis is the accuracy
- The x-axis is the number of operations

**Models:**
- VGG: highest memory and most operations
- AlexNet is very low in computation, but also has poor accuracy and a decent number of parameters
- GoogLeNet: very efficient, but not amazing accuracy
- ResNet: simple design, moderate efficiency, high accuracy

**Key Concepts Developed:**

- Aggressive downsampling
- Trying to maintain computation while increasing accuracy
- Repeated block structures throughout the networks that can be re-used without having to hand-tune parameters.

# Model Ensemble: 2016

The winner of the 2016 challenge just took a bunch of winning models and combined them together into an ensemble classifier. It wasn't very exciting.

The challenge ended in 2017 (now on Kaggle).

[https://www.youtube.com/watch?v=XaZIlVrIO-Q&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r](https://www.youtube.com/watch?v=XaZIlVrIO-Q&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)

# [[Squeeze and Excitation Network]]
![[squeeze-and-excitation-networks.png]]

These add a “Squeeze-and-excite” branch to each residual block that performs global pooling, has fully connected layers, a sigmoid, scaling, and is then added back to the result of the Conv layers. This adds **global context** to each residual block. Showed an improvement over ResNet.

# [[DenseNet|Densley Connected Convolutional Networks]] 
![[AI-Notes/Concepts/cnn-architectures-srcs/Screen_Shot 10.png]]

Dense blocks where each layer is connected to every other layer in a feedforward fashion.
- Different way of doing skip/shortcut connections as we saw in Residual Networks. In Residual Networks, you could propagate gradients better by having an additive shortcut connection.
- These networks concatenate features with later features to re-use the features at different parts of the network.
- Repeat dense block throughout the network.
- Alleviates vanishing gradient, strengthens feature propagation, encourages feature reuse.

# MobileNet: Tiny Networks

These networks are optimized for mobile devices and focus less on getting the absolute best performance and more so on reducing computation, while still getting decent performance. Also uses the idea of repeated blocks.

- Introduces the idea of “Depthwise Convolution” and “Pointwise Convolution”

# Neural Architecture Search (AutoML)

Trying to train one neural network that outputs the architecture for another neural network.

- One network (**controller**) outputs network architecures
- Sample **child networks** from controller and train them
- After training a batch of child networks, make a gradient step on controller network (using **policy gradient**)
- Overtime, controller learns to output good architectures

The first version of this was **extremely expensive**. Each gradient step on a controller requires training a batch of child models. The original paper trained on 800 GPUs for 28 days. Followup work has focused on efficient search.

![[AI-Notes/Concepts/cnn-architectures-srcs/Screen_Shot 11.png]]

The red line shows networks generated by the **controller network**. These got better performance at the same amount of computation as existing networks.

# CNN Architectures Summary

- Early work (AlexNet → ZFNet → VGG) shows that **bigger networks work better**
- GoogLeNet was one of the first to focus on **efficiency** (aggressive stem, 1x1 bottleneck convolutions, global average pool instead of FC layers)
- ResNet showed us how to train extremely deep networks - limited only by GPU memory! Started to show diminishing returns as networks got bigger
- After ResNet: **efficient networks** became central. how can we improve the accuracy without increasing the complexity?
- Lots of **tiny networks** aimed at mobile devices: MobileNet, ShuffleNet, etc.
- **Neural Architecture Search** promises to automate architecture design

### Which Architecture should I use?
- For most problems you should use off-the-shelf architecture
- If you just care about accuracy, **ResNet-50** or **ResNet-101** are great choices
- If you want efficient networks (real-time, run on mobile, etc) try **MobileNets** and **ShuffleNets**