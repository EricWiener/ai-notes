---
tags: [flashcards]
source: [[Strudel_Segmenter_Transformer_for_Semantic_Segmentation_ICCV_2021_paper.pdf]]
aliases: [Segmenter]
summary: semantic segmentation based on the Vision Transformer (ViT) that does not use convolutions, captures global information, and outperforms FCN based approaches.
---

[[Strudel_Segmenter_Transformer_for_Semantic_Segmentation_ICCV_2021_paper.pdf]]
[PyTorch Implementation](https://github.com/rstrudel/segmenter)

# Abstract
- Image segmentation is often ambigious at the level of individual image patches and needs contextual information to reach label consensus.
- Extends [[ViT An Image is Worth 16x16 Words Transformers for Image Recognition at Scale|ViT]] to semantic segmentation.
- Output patches correspond to image patches. Class labels are assigned to each patch either via a [[Linear]] layer or a mask transformer decoder (the latter is better).
- Use models pre-trained for image classification and fine-tune for image segmentation.
- Performance is better for large models and small patches.

# Introduction
- This paper proposes a pure transformer architecture that **captures global context at every layer of the model during the encoding and decoding stages.**
- [[Transformer]]s can capture global interactions between elements of a scene.

![[segmenter-architecture.png]]
**Pipeline**: their pipeline is similar to [[ViT An Image is Worth 16x16 Words Transformers for Image Recognition at Scale|ViT]]:
- Split image into patches and flatten the patches.
- Embed patches via a [[Linear]] layer.
- Pass patches as tokens into transformer encoder.
- The output of the transformer encoder is passed to a transformer decoder that gives per-pixel class scores.
    - The decoder can either be a [[Pointwise Convolution|1x1 conv]]
    - or a transformer-based decoder scheme where learnable class embeddings are procesed jointly with patch tokens to generate patch masks (this yields better results).

# Related Work
- Current SOTA approaches use [[Semantic Segmentation#Idea 4 Fully Convolutional Networks with downsampling upsampling|fully Convolutional Networks with downsampling + upsampling]]. Ex: [[Fully Convolutional Networks for Semantic Segmentation]].
    - Downside: local nature of convolutional filters limits the access to the global information in the image. For semantic segmentation, the labeling of local patches often depends on the global image context.
- DeepLab methods introduce feature aggregation via [[Dilated Convolution]] and spatial pyramid pooling to increase receptive fields and obtain multi-scale features.
- All approaches that use convolutional backbones are biased towards local interactions. The restriction to local operations imposed by convolutions may imply inefficient processing of global image context and suboptimal segmentation results. 

# Approach
- Model is trained end-to-end with a **per-pixel cross-entropy loss**. At inference time, argmax is applied after upsampling to obtain a single class per pixel.

![[segmenter-architecture.png]]
**Encoder**: the encoder is a standard [[ViT An Image is Worth 16x16 Words Transformers for Image Recognition at Scale|ViT]] transformer. The encoder is a transformer encoder consisting of $L$ layers. It maps the inputs $z_0 = \left[z_{0,1}, \ldots, z_{0, N}\right]$ (embedded patches with position encodings) to $\mathbf{Z}_{\mathbf{L}}=\left[z_{L, 1}, \ldots, z_{L, N}\right]$ (contextualized encoding to be used by the decoder).

## Decoder
- The decoder maps the outputs from the encoder $\mathbf{Z}_{\mathbf{L}} \in \mathbb{R}^{N \times D}$ to a segmentation map $\mathbf{s} \in \mathbb{R}^{{H} \times W \times K}$  where $K$ is the number of classes. 
- It first maps patch-level encodings to patch-level class scores. Then, the patch-level class scores are upsampled via bilinear interpolation to pixel-level scores.

### Linear:
- The linear decoder is just a linear layer with $D$ inputs and $C$ outputs: `nn.Linear(self.d_encoder, n_cls)`.
- It is applied to the $\mathbf{Z}_{\mathbf{L}} \in \mathbb{R}^{N \times D}$ outputs of the encoder and results in $\mathbf{Z} _{\operatorname{lin}} \in \mathbb{R}^{N \times K}$
- The sequence is then reshaped to be $\mathbf{S}_{\operatorname{lin}} \in \mathbb{R}^{H / P \times W / P \times K}$ and then bilinearly upsamped to the original image size $\mathbf{S} \in \mathbb{R}^{H \times W \times K}$
- Softmax is then used to get the final segmentation map.

### Mask Transformer
- Use $K$ learnable class embeddings where $K$ is the number of classes. $\textbf{cls} = \left[\operatorname{cls}_{1}, \ldots, \mathrm{cls}_{K}\right] \in \mathbb{R}^{K \times D}$ where $D$ is the same dimension as the encoder outputs/inputs to the decoder.
    - Each embeddings is randomly initialized and corresponds to a single semantic class.
- The class embeddings $\textbf{cls}$ are jointly initialized with the patch embeddings ($\mathbf{z}_{\mathbf{L}}$) from the encoder.
- The decoder is a transformer encoder composed of $M$ layers. It outputs patch embeddings $\mathbf{z}_{\mathbf{M}}^{\prime} \in \mathbb{R}^{N \times D}$ and class embeddings $\mathbf{c} \in \mathbb{R}^{K \times D}$.
- For each of the $N$ patches, $K$ masks (one per class) are computed via the scalar product between the L2-normalized patch embeddings and the class embeddings. $\operatorname{Masks}\left(\mathbf{z}_{\mathbf{M}}^{\prime}, \mathbf{c}\right)=\mathbf{z}_{\mathbf{M}}^{\prime} \mathbf{c}^{T}$ where $\operatorname{Masks}\left(\mathbf{z}_{\mathbf{M}}^{\prime}, \mathbf{c}\right) \in \mathbb{R}^{N \times K}$.
- The masks are then reshaped from $N \times K$ to $H / P \times W / P \times K$ and bilinearly upsamples to the original image size of $H \times W \times K$.
- A softmax is then applied on the class dimension followed by a layer norm to obtain pixel-wise class score forming the final segmentation map. The individual pixels in the segmentation map are all exclusive to each other: $\sum_{k=1}^{K} s_{i, j, k}=1 \text { for all }(i, j) \in H \times W$

# Implementation
**Transformer encoder**
- Used [[ViT An Image is Worth 16x16 Words Transformers for Image Recognition at Scale|ViT]] models of varying sizes. The only parameters considered were the number of layers and token size.
- The head size of a [[Attention#Multihead Self-Attention Layer|multiheaded self-attention block]] is fixed to 64
- The number of heads is the token size / the head size
- The hidden size of the linear layer following the multiheaded self-attention layer is four times the token size.
- Also consider different size patches (8x8, 16x16, 32x32).
- All models are initialized via the improved ViT models from [[How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers]].

**Transformer decoder**
- The mask transformer has 2 layers with the same token and hidden size as the encoder.

**Data Augmentation**:
- Follow the standard pipeline from MM-Segmentation: which does mean substraction, random resizing of the image to a ratio between 0.5 and 2.0 and random left-right flipping.

**Optimization**:
- Fine-tune the pre-trained models with pixel-wise cross-entropy loss without weight rebalancing.
- Use SGD with no weight decay and a "poly" learning rate decay.

**Regularization**:
- CNNs usually use [[Research Papers/Batch Normalization]] which acts as a regularizer. Transformers usually use [[Layernorm|layer normalization]] with dopout as a regularizer during training.
- [[Dropout]]: had a negative affect on performance, so it wasn't used.
- [[Stochastic Depth]]: randomly skip a layer of the transformer. All models were trained with stochastic depth set to 0.1.

**Patch size**:
- Increasing patch size results in a coarser representation of the image but also a smaller sequence that is faster to process.
- Reducing the patch size leads to improvements that doesn't require any additional parameters, but requires attention over longer sequences (increasing compute and memory usage).
- A patch size of 32 results in globally meaningful segmentation, but produces poor boundaries.
- Reducing the patch size leads to considerably sharper boundaries.

# Results
- The largest model with smallest patch size outperforms SOTA convolutional approaches.
- Seg/16 models perform best in terms of accuracy versus compute time. Seg-B-Mask/16 offers a good trade-off and outperforms FCN based approaches with similar inference speed and four times faster inference than ResNet-50 which providing similar performance.
- They observed a significant drop in performance when the training size is below 8k images.
- Deeplabv3+ tends to generate sharper object boundaries while Segmenter provides more consisent labels on large instances and handles partial occlusions better.
- Using a mask transformer as a decoder yields better results than using a linear decoder.