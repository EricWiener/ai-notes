---
tags: [flashcards]
source:
summary: introduces the Dense Convolutional Network (DenseNet), which connects each layer to every other layer in a feed-forward fashion. Paper from 2016
---
[YouTube Summary](https://youtu.be/hSC_0S8Zf9s)

For each layer, the feature-maps of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers.

### Advantages
DenseNets alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters.

**Why do DenseNets help with the vanishing gradient problem?**
??
Adding shortcut connections will result in a better gradient flow.
<!--SR:!2024-05-19,482,309-->

**Why do DenseNets strengthen feature propagation /  gradient flow?**
??
Features from previous layers are passed to all later layers.
<!--SR:!2025-08-16,762,270-->

**Why do DenseNets substantially reduce the number of parameters?**
??
Because of how [[DenseNet#Forward Propagation]] works, there is a much slower growth than traditional CNNs which often have growth like 128, 256, 512 channels.
<!--SR:!2024-11-26,614,288-->

For a ResNet, if you have $C$ input channels and produce $C$ output channels, you will need to use a $3 \times 3 \times C$ kernel $C$ times (need to apply it to all $C$ input channels to produce one output layer, so you need to apply it $C$ times to produce $C$ outputs). This results in $O(C \times C)$ parameters.

For a DenseNet, you have $\ell \times k$ inputs and $k$ additional outputs. You therefore have a $3 \times 3 \times (\ell \times k$) kernel that you need to apply $k$ times (or $5k$ times in the case of DenseNet-B). This is $O(\ell \times k \times k$) parameters.

$k \ll C$, so DenseNet will use fewer parameters ($C$ is like 256, 512, 1024 and $k$ is like 12 or 32).


### [[ResNeXt#Skip Connection]]
- ResNets have the following transformation function: $\mathbf{x}_{\ell}=H_{\ell}\left(\mathbf{x}_{\ell-1}\right)+\mathbf{x}_{\ell-1}$ where $H_\ell$ is a transformation function an $x_{\ell - 1}$ is the feature map from the previous layer. ResNets add the skip connection via $+ x_{\ell - 1}$
- An advantage of ResNets is that the gradient can flow directly through the identity function from later layers to the earlier layers.
- However, the identity function and the output of each layer are combined by summation, which may impede the information flow in the network.

### Forward Propagation
During forward propagation you can decide on how many new output channels to generate. For instance, if the previous layer had 3 channels and you generated 10 new output channels (10 is referred to as the ==growth rate== and is the number of additional layers you add at each step), you would then concatenate the 3 channels with the new 10 channels to get 13 total channels. You then repeat this for the next layer and get 23 total channels (10 of which are new). 
<!--SR:!2025-05-19,772,328-->

![[densenet-20220721113601762.png|500]]
![[densenet-20220721113608540.png|500]]
If you have $\ell$ layers and a growth rate of $k = 10$ and then you will end up with $\ell * k$ layers.

**Composite Layer**
The Composite Layer ==takes the concatenated feature maps and produces the new feature maps (the number of which are determined by the growth rate)==.
<!--SR:!2025-03-01,490,269-->

DenseNet's Composite Layer is denoted as $\mathbf{x}_{\ell}=H_{\ell}\left(\left[\mathbf{x}_{0}, \mathbf{x}_{1}, \ldots, \mathbf{x}_{\ell-1}\right]\right)$ where $\left[\mathbf{x}_{0}, \mathbf{x}_{1}, \ldots, \mathbf{x}_{\ell-1}\right]$ refers to refers to the concatenation of the feature-maps produced in layers 0, . . . , $\ell - 1$ and $H_{\ell}$ is a composite function of three individual operations: **BatchNorm, ReLU, and a 3x3 Conv**.

**Composite Layer with Bottleneck Layer**
Each layer only has $k$ (growth rate) additional outputs, but it often has many more inputs. To increase computational efficiency, you can add a ==1×1 convolution== before each 3×3 convolution to reduce the number of input feature-maps to the 3x3 conv.
<!--SR:!2023-11-29,310,268-->

The BN-ReLU-Conv(1× 1)-BN-ReLU-Conv(3×3) version of $H_{\ell}$ is called DenseNet-B. The paper lets each 1x1 convolution produce a $4k$ feature map. This is then passed to the 3x3 conv which produces a $k$ feature map (which is then concatenated to the previous feature maps and passed onto the next layer). This makes it so the 3x3 conv only has to operate on a  feature map with $4k$ channels vs. the larger number of channels that the $1 \times 1$ conv operates on.

**Transition Layer**
![[densenet-20220722073007015.png]]

The DenseNet architecture doesn't connect all the previous feature maps throughout the network because concatenating tensors requires their height and width be the same. Instead, it concatenates feature maps within Dense Blocks and then has Transition Layers between Dense Blocks.

An essential part of convolutional networks is ==down-sampling layers that change the size of feature-maps==. The Transition Layers achieve this with a BatchNorm, 1x1 conv (decrease number of channels), and then a 2x2 average pooling (decrease spatial size).
<!--SR:!2024-12-16,599,308-->

**Compression**
- To improve model compactness, the transition layer can ==reduce the number of channels==.
- If a dense block contains $m$ feature maps, the following transition layer will generate $\lfloor \theta m \rfloor$ feature maps where $\theta$ ($0 < \theta \leq 1$) is called the **compression factor**.
- DenseNet with $\theta < 1$ is called DenseNet-C (and $\theta = 0.5$ is used in the paper).
- DenseNet with both $\theta = 0.5$ and the bottleneck layer added is called DenseNet-BC.
<!--SR:!2024-07-29,545,328-->


### Summation vs. Concatenation
- Traditional feed-forward architectures can be viewed as algorithms with a state, which is passed on from layer to layer. Each layer reads the state from its preceding layer and writes to the subsequent layer. It changes the state but also passes on information that needs to be preserved.
- The summation of previous feature maps is done in ResNet via the skip connection. This makes the preservation of information explicit through the identity transformation. Adding the previous feature map to the outputs of the current layer will cause the features to become correlated with each other. Concatenation allows the features to remain diversified.
- The concatenation of tensors was first used in [[Inception Networks]] and later in DenseNet. Compared to Inception networks, which also concatenate features from different layers, DenseNets are simpler and more efficient
- Concatenation allows the next layer to choose to work on either the feature maps generated by the immediate earlier layer or it can work on the feature maps of the conv operations that took place early.
- DenseNet adds only a small set of feature-maps to the ==“**collective knowledge**”== of the network and keep the remaining feature maps unchanged. The final classifier makes a decision based on all the feature maps in the network.
- Concatenating feature-maps learned by different layers increases variation in the input of subsequent layers and improves efficiency.
<!--SR:!2024-04-17,450,289-->

### Architecture
![[densenet-architecture.png]]

### Model Types
- DenseNet: plain DenseNet
- DenseNet-B: DenseNet with BottleNeck layer added in each DenseBlock (1x1 conv).
- DenseNet-C: DenseNet with compression factor $\theta < 1$ in the transition layer.
- DenseNet-BC: combination of DenseNet-B and DenseNet-C.

### Performance
- DenseNets achieve lower error rates while using fewer parameters than ResNet. Without data augmentation, DenseNet performs better by a large margin

![[densenet-bc-vs-resnet-chart.png]]
- You can use three times fewer parameters as ResNet and achieve the same performance using DenseNet-BC.

### Advantages
- Works well with less training data
- Results in smoother decision boundaries

### Memory Efficient DenseNet
[This](https://ar5iv.labs.arxiv.org/html/1707.06990) paper implements a memory efficient DenseNet that uses shared memory storage to avoid needing to keep creating new tensors of the previous outputs + the new $k$ additional channels.
![[memory-efficient-densenet.png|700]]


