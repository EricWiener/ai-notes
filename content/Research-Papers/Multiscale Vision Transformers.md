---
tags: [flashcards]
source:
aliases: [MViT]
summary: introduce multiscale feature hierarchies to transformer models.
---

### Abstract
- Add multiscale feature hierarchies to transformer models using several channel-resolution scale stages.
- Stages expand the channel capacity while reducing spatial resolution. Each stage consists of multiple transformer blocks with specific space-time resolution and channel dimension.
- Creates a multiscale pyramid of features with early layers using high spatial resolution to model simple low-level visual information and deeper layers being spatially coarse but complex, high-dimensional features. This is done by expanding the channel capacity while pooling the resolution from input to output of the network.
- Main focus of the paper is performance on video recognition. Other papers that work on video recognition with transformers need 5x more computation and large-scale external pre-training on ImageNet-21K to achieve similar accuracy.

### Motivating history
Neural networks for visual processing pattern:
- Reduction in spatial resolution as one goes up processing hierarchy.
- Increase in number of different channels with each channel corresponding to specialized features.

Computer vision multiscale processing ([[Object Detection#Image Pyramid]]):
- Process images at multiple scales to handle features at different scales.
- Decreased computing requirements by working at lower resolution.
- You could have more "context" at lower resolutions which could then guide higher resolutions.

### Performance on videos
- Vision transformers ([[ViT An Image is Worth 16x16 Words Transformers for Image Recognition at Scale]]) trained on natural video don't suffer a performance decay when tested on videos with shuffled frames. This means the models aren't using temporal information and rely heavily on appearance.
- MViT models tested on shuffled frames observe significant accuracy decay which indicates they are using temporal information.

### Related work
- [[Training data-efficient image transformers & distillation through attention|DeiT]] proposes a data efficient approach to training [[ViT An Image is Worth 16x16 Words Transformers for Image Recognition at Scale|ViT]]. MViT's training recipe builds on this approach.

# Multi Head Pooling Attention (MHPA)
![[multi-head-pooling-attention.png|450]]

- In contrast to [[Attention#Multihead Self-Attention Layer]] (MHA), MHPA pools the sequence of latent (intermediate/hidden) tensors to reduce the sequence length (resolution) of the attended input.
- You first project $X$ through three different Linear layers and then get intermediate $\hat{Q}, \hat{K}, \hat{V}$ of dimensions $T H W \times D$.
- These are then independently pooled by first unflattening, pooling, and then flattening again.
- The keys and values are pooled to shape $\widetilde{T} \widetilde{H} \widetilde{W} \times D$. The query is pooled to shape $\hat{T} \hat{H} \hat{W} \times D$. The output of the layer will have the same shape as the query ($\hat{T} \hat{H} \hat{W} \times D$). The shape of the key and value vectors affects computation and memory, but does not affect size of the output.


> [!NOTE] The query and keys/values are pooled to different dimensions. The output will have the same shape as the query. See the color-coded shapes in the diagram above.


**Multiple heads**: the computation can be parallelized with $h$ heads where each head performs the pooling attention on a non overlapping subset of $D/h$ channels of the $D$ dimensional input tensor $X$.

# Multiscale Vision Transformers Architecture
- The key concept is to grow the channel resolution (dimension) while reducing the sequence length (spatiotemporal resolution) throughout the network. In early layers there is fine spacetime (sequence length) and coarse channel resolution (few channels) in early layers. In later layers there will be a shorter sequence length (coarse spacetime) and a finer channel resolution (more channels).
- The model is built up up **scale stages** which are a sequence of $N$ transformer blocks that operate on the same scale with identical resolution across channels and space-time dimensions.
- At **stage transition** (moving between scale stages) the channel dimension of the processed sequence is upsampled while the length of the sequence is downsampled.

### Channel Expansion
- When transitioning from one stage to the next, the channel dimension will be expanded by increasing the output of the final MLP layer in the previous stage.
- In the implementation, when downsampling the spacetime resolution by 4x, the channel dimension was increased by 2x.

### Query Pooling
- Since the paper decreases resolution at the start of the stage and then keeps resolution throughout a stage, the query is only pooled in the first layer of each stage.

### Key-Value Pooling
- Unlike Query pooling, changing the sequence length of key K and value V tensors, does not change the output sequence length and, hence, the space-time resolution. However, they play a key role in overall computational requirements of the pooling attention operator.
- Key/Value pooling occurs at every layer (unlike Query pooling that occurs at the first layer only).
- The sequence length (and hence the stride used when pooling) needs to be identical for keys/values.

### Skip Connections
Since the channel dimension and sequence length change inside a residual block, we pool the skip connection to adapt to the dimension mismatch between its two ends.

The pooled input $X$ is added to the output (vs. using the raw input $X$).