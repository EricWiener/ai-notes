---
tags: [flashcards]
aliases: [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation]
source: 
summary: use an encoder-decoder structure with spatial pyramid pooling with an Inception backbone (vs. ResNet). They use depthwise separable convolutions in the ASPP and decoder modules.
---

[YouTube Video](https://youtu.be/Gzrej8ciK9o)

# Introduction
- [[DeepLab#Atrous Spatial Pyramid Pooling (ASPP)|ASPP]] encodes multi-scale contextual information by using multiple convolutions with different dilation rates to look at the features with different FOVs.
- [[Encoder-Decoder Models|Encoder-Decoder]] networks capture sharper object boundaries by gradually recovering information from the encoder in the decoder.
- DeepLabv3+ uses [[DeepLabv3]] as an encoder and adds a decoder module to get more detailed object boundaries using information from the encoder. They also explore the Xception model and apply [[Depthwise Separable Kernels]] for both the ASPP and decoder modules.

![[deeplabv3-plus-fig-1.png]]
DeepLabv3+ uses the ASPP from DeepLabv3 (a) with an encoder-decoder structure (b) to create DeepLabv3+ (c). Using dilated convolutions in the encoder module allows extracting features at an arbitrary resolution and allows you to trade off between precision and runtime.

They use an encoder-decoder structure because it is too computationally expensive to extract high resolution feature maps with current GPU memory limitations and the designs of state-of-art neural networks like ResNet-101. Using an encoder-decoder model is beneficial since no features are dilated in the encoder path to preserve resolution (as was needed to be done in [[DeepLabv3]]) and the sharp object boundaries can be recovered in the decoder path.

# Architecture
### Dilated Depthwise Convolution
[[Depthwise Separable Kernels]] reduce the computation and number of paramters while maintaining similar (or slightly better) performance.

![[dilated-depthwise-convolution.png]]
A 3 × 3 depthwise separable convolution decomposes a standard convolution into (a) a **depthwise convolution** (applying a single filter for each input channel) and (b) a **pointwise convolution** (combining the outputs from depthwise convolution across channels). In this work, we explore **atrous separable convolution** where atrous convolution is adopted in the depthwise convolution, as shown in (c) with rate = 2.

They saw the atrous seperable convolution significantly reduces the computation complexity of proposed model while maintaining similar (or better) performance.

![[deeplabv3-plus-architecture.png]]
### Encoder
The last feature maps before logits in [[DeepLabv3]] (only chopping off last $1 \times 1$ conv that produces num_class channels) is used as the encoder output in the encoder-decoder structure. The encoder feature map contains 256 channels and lots of semantic information. Since the encoder uses dilated convolutions, you could extract features at an arbitrary resolution.

### Decoder
**DeepLabv3's naive decoder**
In [[DeepLabv3]], the features produced by the ASPP module are `[B, num_classes, H/16, W/16]`. The features are then bilinearly upsampled by a factor of 16 to `[B, num_classes, H, W]` to produce the final outputs. This can be considered a naive decoder and it doesn't do a good job of recovering object boundaries.

**Improved decoder:**
- The encoder features are bilinearly upsampled by a factor of 4.
- You take low-level features from the network backbone feature map of the same spatial resolution as the encoder features (ex. Conv2 before striding in ResNet-101). 
- You then pass the features from the backbone through a 1×1 conv to reduce the number of channels since the low-level features in the backbone usually contain a large number of channels (e.g. 256 or 512) that may outweight the importance of the encoder features (e.g. 256 channels in this model) and make training harder.
- You then concatenate the encoder features and the backbone features that have passed through the 1x1 conv.
- You pass the concatenated features through a couple $3 \times 3$ convolutions.
- You then upsample the resulting features by another factor of 4 to be at the same resolution as the input image.

Using `output_stride = 16` for the encoder module yields the best trade-off between speed and accuracy (`output_stride = 8` only has a marginal accuracy improvement with a lot of extra computation).

### Modified Xception
DeepLabv3+ experiments with using [[Xception]] as the backbone instead of ResNet and they see improved performance. Xception replaces the Inception modules in [[Going Deeper with Convolutions|Inception Networks]] with [[Depthwise Separable Kernels]]. DeepLabv3+ further modifies [[Xception]] by using dilated convolutions. They make the following modifications:
1. Use a deeper network (except they leabe the entry flow unmodified for fast computation and memory efficiency).
2. All max-pooling ops are replaced by depthwise separable convolutions with striding (which enables using dilated separable convolutions to extract feature maps at any resolution).
3. Add [[Research Papers/Batch Normalization]] and [[ReLU]] activation after each $3 \times 3$ depthwise convolution.

![[xception-network-architecture.png|400]] ![[deeplabv3-plus-modified-xception.png|400]]
On the left is the original Xception architecture and on the right is the modified architecture.

# Ablations
### Decoder Ablations
![[deeplabv3-plus-architecture-annotated.png]]
They consider three places for different design choices, namely (1) the 1 × 1 convolution used to reduce the channels of the low-level feature map from the encoder module, (2) the 3 × 3 convolution used to obtain sharper segmentation results, and (3) what encoder low-level features should be used.

**The 1 × 1 convolution used to reduce the channels of the low-level feature map from the encoder module:**
Reducing the channels of the low-level feature map from the encoder module to either 48 or 32 leads to better performance. The use $[1 \times 1, 48]$ for channel reduction (1x1 conv with 48 filters).

**The 3 × 3 convolution used to obtain sharper segmentation results:**
It is better to use two $3 \times 3$ convolutions with 256 filters than using one or three convolutions. Changing the number of filters from 256 to 128 or the kernel size from $3 \times 3$ degrades performance.

**Which encoder backbone features are used:**
They tried using both `conv2` and `conv3` features but didn't see any significant improvements.

### Backbone Ablations
- Using Xception over ResNet-101 gave a performance boost.
- Using depthwise seperable convolutions in the ASPP and decoder modules maintained similar mIOU performance with a 33% reduction in Multiply-Adds.

# Results
### Object Boundaries
Using the decoder (over naive upsampling) improved object boundary predictions. They evaluate segmentation accuracy using the approach from [[Robust higher order potentials for enforcing labelconsistency#Accuracy Evaluation]]. The performance improvement over the naive decoder (upsampling) when the [[Dilation|dilation]] band is narrow (only consider the area closer to the object border).

![[seg-results-w-naive-decoder-and-w-decoder.png]]
The above shows the original image, the segmentation results with naive upsampling, and the results with the decoder module.