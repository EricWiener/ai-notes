---
tags: [flashcards]
aliases: [Deep Residual Learning for Image Recognition, ResNet18, ResNet34, ResNet101, ResNet152]
source: https://ar5iv.labs.arxiv.org/html/1512.03385
summary:
---
This was the 2015 winner of the ImageNet challenge. This network had 152 layers, which was a significant jump from previous years ([[Going Deeper with Convolutions|GoogleNet]] was only 22), but it also got half the error rate of Once Batch Normalization was discovered, people were able to train much deeper networks. 

![[AI-Notes/Concepts/cnn-architectures-srcs/Screen_Shot 4.png]]

However, these deeper models didn't perform well on test error, so people at first thought they were overfitting the training data. However, it also performed poorly on the training data, so it actually is **underfitting.**

A deeper model should have the ability to emulate a shallower model. You could copy all the layers from the shallower model and set the remaining layers to the identity.

> [!note]
> Therefore, deeper models should be able to do at least as good as shallow models.
> 

This was an **optimization** problem. Deeper models are harder to optimize and in particular don't learn identity functions to emulate shallow models. The solution is to change the network so learning identity functions is easier in case they had too many layers.

# Residual Networks
Residual networks are a way to make it easier to learn identity functions.

![[AI-Notes/Concepts/cnn-architectures-srcs/Screen_Shot 5.png]]
![[AI-Notes/Concepts/cnn-architectures-srcs/Screen_Shot 6.png]]
Note that you add $x_i$ to the output of the second convolutional layer before passing to ReLU

You add the input $X$ to the output of the second CONV layer. This means the overall block (includes both CONV layers) actually computes $F(X) + X$. If you set $F(X) = 0$, then you will just end up with the identity function. This should make it easier for deeper networks to emulate shallower networks. Also, when back-propagating through an add, it will copy the gradient along both edges of the computational graph. These helps improve the propagation of gradient information.

[[Backpropagation]]

A residual network is a stack of many residual blocks. It combines principles of VGG and GoogLeNet. It uses a regular design like VGG: each residual block has two 3x3 CONV. The network is divided into stages. The first block of each stage halves the resolution (with stride-2 CONV) and doubles the number of channels.

Note that for the residual blocks where the first convolution has stride=2 and doubles the number of channels, the skip connection is replaced with a 1x1 convolution with a stride and number of channels necessary to make sure that the output of the 1x1 convolution has the same shape as the output of the second convolution layer of the block. For all other layers, the shape of the input matches the shape of the output of the second layer so this isn’t an issue.

![[Screenshot_2022-02-02_at_07.47.362x.png]]
Note that the first orange, blue, and yellow conv blocks has a stride of 2 (denoted with /2) to halve the resolution and double the number of channels.

It used the aggressive **stem** as GoogLeNet to aggressively downsample the input, as well as the idea of **global average pooling** and a single linear layer at the end.

You just need to choose the number of residual blocks per stage and the size of the original input. 

**ResNet-18:**
- Stem: 1 conv layer
- Stage 1 (C=64): 2 res. block = 4 conv
- Stage 2 (C=128): 2 res. block = 4 conv
- Stage 3 (C=256): 2 res. block = 4 conv
- Stage 4 (C=512): 2 res. block = 4 conv
- Linear

This is 18 total layers.

# Bottleneck Block
This replaces the basic residual block.

![[AI-Notes/Concepts/cnn-architectures-srcs/Screen_Shot 7.png]]

- It accepts an input tensor with four times as many channels as the basic block.
- The first layer is a 1x1 convolution that reduces the number of channels from 4C to C
- Then we perform a 3x3 convolution
- Then we have another 1x1 convolution that expands the number of channels again

This has the benefit of reducing the number of channels before convolving with a larger filter. This has more layers and less computational cost. You add non-linearity between layers, which results in more nonlinear computation. Deeper layers with more non-linearity should be able to create more complex functions. 

ResNet-50 replaces all basic blocks with bottleneck blocks and performs very well.

> [!note]
> ResNet won every track in the ImageNet challenge, as well as every challenge in COCO (competition run by Microsoft).
> 

# Architectures
![[resnet-architectures.png]]
ResNet18 and 34 both use `BasicBlock`. ResNet50, 101, and 152 all use `BottleneckBlock`s which first use a 1x1 convolution to decrease the number of channles, use a 3x3 convolution, and then use a 1x1 convolution to increase the number of channels again.

The spatial resolution is decreased for all variants via `stride=2` in the **first 3x3 conv** in the **first block**. For the ResNet18,34 variants, this is the initial `Conv2D` layer and for the 50, 101, and 152 variants this is the 3x3 `Conv2D` layer (follows the 1x1 layer). [See code here](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py).

**Naming of layers**
Note that `conv3_1` refers to the initial `BottleneckBlock` for the third layer (i.e. it doesn't correspond to the 1x1 convolution inside the block).

**Naming in table vs. code**
In many implementations, `conv1` is typically referred to as `conv1` and then the layers are considered as only `conv2_x`-`conv5_x` in the table above. For instance, here is the `forward()` pass from [PyTorch's implementation](https://github.com/pytorch/vision/blob/474d87b8a942eb0d5f22f5d98120c6f8961c798e/torchvision/models/resnet.py#L266): 
```python
def _forward_impl(self, x: Tensor) -> Tensor:
    x = self.conv1(x) # conv1 in table above
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    
    x = self.layer1(x) # conv2
    x = self.layer2(x) # conv3
    x = self.layer3(x) # conv4
    x = self.layer4(x) # conv5
```
Note how there is a mismatch between the layer numbers and the numbers in the table.

# Improving Residual Blocks: Pre-Activation ResNet Block
![[pre-activation-resnet-block.png]]

The “Pre-Activation” ResNet Block puts the ReLU inside the residual block. Unlike the original block where the input is added to the output of the residual block via the skip connection and then goes through the ReLU, here you don’t pass the input through the ReLU. This lets you learn a true identity function. It saw a slight improvement in performance, but it **is not used much in practice.**

# G Parallel Pathways
Use multiple bottleneck residuals in parallel. It was used in [[ResNeXt]] which was an improvement of ResNet. It has G parallel pathways where each pathway is a bottleneck block. It took inspiration from GoogleNet (hence why **G** parallel pathways). 

The channel dimension of each of the pathways is a new parameter $c$.

After you compute the outputs of each of the pathways, they are all added together. 

![[AI-Notes/Concepts/cnn-architectures-srcs/Screen_Shot 8.png]]

You can manipulate $c$ to get the parallel pathways to have the same cost as the original ResNet. **This adds an additional parameter where you can modify the network.** In PyTorch `grouped convolution` does the same concept.

ResNeXt uses [[Grouped Convolutions]] in the middle part of the bottleneck block to get the same affect as having parallel pathways (it doesn’t actually have parallel pathways).

> [!note]
> While maintaining computational complexity by manipulating $c$, you can add more groups to get **higher performance with the same computational complexity.**
> 