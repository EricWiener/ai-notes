---
tags: [flashcards]
source:
summary: this describes how to add attention blocks to vision models. TLDR is you divide an image into patches, flatten the patches into a sequence, and then compute attention using this sequence. Models like Swin will shift windows around to share patch information.
---

### Idea 1: add attention to existing CNNs
![[adding-attention-to-cnns.png]]
- This was done by [[Non-Local Neural Networks]]
- Start from a standard CNN architecture (ex. ResNet) and add self-attention blocks between existing ResNet blocks.
- The model is still a CNN, but with additional modules.

### Idea 2: replace convolution with "local attention"
Examples:
- [[Local Relational Networks for Image Recognition]]
- [[Stand-Alone Self-Attention in Vision Models]]

**Typical [[Convolutional Neural Networks|CONV layer]]**:
![[typical-conv-layer.png]]
In a typical conv layer, the output at each position is the inner product of conv kernel with receptive field in input. This takes advantage of the spatial relationship of the data since you just consider a local area.

**Local Attention**:
![[local-attention-center-is-query.png]]
In local attention, you map the center of each receptive field to a **query** vector of dimension $D_Q$ (you can transform the value from the feature map via a 1x1 conv to get a $D_Q$ dimension vector).

![[local-attention-key-values.png]]
You then map each element in the receptive field to a **key and value** vector. The keys are of dimension $D_Q$ and the values are of dimension $C'$ (the same dimension as the output). You can then compute one element of the output using [[Attention]] (note that you don't use self-attention here, just regular attention, since there is only one query with multiple keys/values).

You can then replace all the convolutions with local attention.

**Local Attention Analysis**:
- Tricky implementation details and only marginally better than ResNets
- Did not take off as the authors had intended it to.
- You have to re-compute the dot product between each query vector and the key vectors for every location. You will end up attending to the same key vectors multiple times for different queries. 

### Idea 3: Standard Transformer on Pixels
Ex: [[Generative Pretraining from Pixels]]

![[standard-transformer-on-pixels.png]]

Transformers work great on text, so it was attempted to use them similarly on images. The idea here is that you can treat an image as a set of pixel values and then feed the input to a standard Transformer. Now, each pixel attends to every other pixel and you output a feature vector for every pixel.

You can then try to do things like generate new images by training a model with masking pixels and try to predict the missing pixels.

**Analysis**:
- This approach uses too much memory.
- An $R \times R$ image needs $R^4$ elements per attention matrix (you have $R^2$ total pixels in the image and each pixel needs to attend to every other pixel, so this is $((R^2)^2 = R^4$).
- R=128, 48 layers, 16 heads per layer takes 768GB of memory for attention matrices for a single example (you would need 10 H100 GPUs for one image).

# Idea 4: Standard Transformer on Patches
Example: [[ViT An Image is Worth 16x16 Words Transformers for Image Recognition at Scale|ViT]]
![[vit-overview-diagram.png]]

![[vit-overview-eecs498.mp4]]

### ViT improvements via data augmentation and regularization
[[How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers]] showed regularization and data augmentation could improve [[ViT An Image is Worth 16x16 Words Transformers for Image Recognition at Scale|ViT]] performance.

Regularization: 
- Weight Decay
- Stochastic Depth
- Dropout (in FFN layers of Transformer)

Data Augmentation:
- MixUp
- RandAugment

### ViT improvements via [[Distillation]]
This approach was presented in [[Training data-efficient image transformers & distillation through attention]]. It showed significant improvements distilling a ViT from a [[ResNet]].

- You first train a teacher CNN
- You can then train a student [[ViT An Image is Worth 16x16 Words Transformers for Image Recognition at Scale|ViT]] to match the predictions from the teacher CNN. 

![[adding-distillation-token-to-vit.png]]
To use distillation with ViT, the paper adds a **distillation token** which (like the **classification token**) is a learned embedding that is appended as an input to the transformer. The predicted class scores of the distillation token should match the class scores of the teacher. The predicted class scores from the classification scores should match the ground truth. These tokens are randomly initialized and then updated via gradient descent.

Note: you can just add extra inputs to a transformer because it operates on sets.

# ViT vs. CNN
**CNNs**:
- In most CNNs, they decrease the resolution and increase the number of channels as you go deeper in the network. This is useful since objects in images can occur at various scales.
- Features at the earlier layers of the network are higher resolution and look at smaller areas of the input image.

**ViT**:
- All blocks have the same resolution and number of channels (same number of patches and resolution). These type of architectures are called [[Isotropic Architectures]].
- You can build a hierarchial ViT model using the [[Swin Transformer]].