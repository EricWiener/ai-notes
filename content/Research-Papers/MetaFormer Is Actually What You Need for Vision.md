---
tags: [flashcards]
source: https://arxiv.org/abs/2111.11418
summary: the general architecture of the Transformers, instead of the specific token mixer module, is more essential to model performance. Replace the attention module in Transformers with a simple spatial pooling operator to conduct only basic token mixing.
---

[Code](https://github.com/sail-sg/poolformer)

### Abstract
- A common belief is that the [[Attention|attention]]-based token mixer module contributes most to the success of [[Transformer|Transformers]].
- This work shows that the general architecture of the Transformer (vs. the specific token mixer module) is more essential to the model's performance.
- This paper makes a new model, PoolFormer, that replaces attention with spatial pooling for basic token mixing. The new model is competitive with SOTA models.
- Paper describes a new concept of ==MetaFormer== which is a general architecure abstracted from [[Transformer|Transformers]] without specifying the token mixer.
<!--SR:!2025-06-29,806,330-->

# Introduction
### [[Transformer]]
Consists of two main components:
- Attention module for mixing information among tokens (called the ==token mixer==).
- The other component contains the remaining modules such as channel MLPs and residual connections.
<!--SR:!2024-10-26,618,330-->

[[MLP-Mixer]] replaces the attention module with spatial MLPs (mixes spatial info - not channel info) as token mixers and finds the MLP-like model is competitive on image classification benchmarks.

### MetaFormer
- The MetaFormer abstracts the token mixing module (the token mixer is still used, but it doesn't have to be a specific type, e.g. attention).
- The main function of the token mixer is to ==mix spatial token information (mix the data within each channel)== although some token mixers can also mix channels, like attention.
- The success of Transformer/MLP-like models is largely attributed to the MetaFormer architecture (not attention).
<!--SR:!2024-01-07,278,252-->

### PoolFormer
- Replaces attention with pooling (non-parametric operator used for demonstration purposes).
- Achieves competitive performance with SOTA models using sophisticated token mixers.
- The pooling operator has no learnable parameters and it just makes each token averagely aggregate its nearby token features.
- Like [[Swin Transformer]], PoolFormer uses a hierarchial architecture (number of channels increase at each stage and spatial dimensions are reduced).

**Pooling Layer for PoolFormer**:
- Since the MetaFormer block already has a residual connection, subtraction of the input itself is added for the Pooling Layer.
- The pooling layer computes an `AvgPool2d` with stride=1, pool_size=3, and padding=1.
```python
import torch.nn as nn

class Pooling(nn.Module):
   def __init__(self, pool_size=3):
      super().__init__()
      self.pool = nn.AvgPool2d(
         pool_size, stride=1,
         padding=pool_size//2,
         count_include_pad=False,
      )
   def forward(self, x):
      """
      [B, C, H, W] = x.shape
      Subtraction of the input itself is added
      since the block already has a
      residual connection.
      """
      return self.pool(x) - x
```

**Pooling vs. other token mixers:**
- Self-attention and spatial MLP have computational complexity quadratic to the number of tokens to mix. 
- Spatial MLPs bring much more parameters when handling longer sequences.
- Self-attention and spatial MLPs usually can only process hundreds of tokens.
- Pooling needs a computational complexity linear to the sequence length without any learnable parameters.
- With the pooling operator, each token evenly aggregates the features from its nearby tokens (vs. a learned approach).

**PoolFormer [[Layernorm|layer normalization]]**
- Modified Layer Normalization to compute the mean and variance along token and channel dimensions compared to only channel dimension in vanilla Layer Normalization

**Other details**
- Dropout is disabled but stochastic depth [27] and [[LayerScale]] are used to train deeper models.

# Results and Analysis
- PoolFormer is highly competitive with other CNNs and MetaFormer-like models.
- Since the local spatial modeling ability of the pooling layer is much worse than the neural convolution layer, the competitive performance of PoolFormer can only be attributed to its general architecture MetaFormer.
- MetaFormer is actually what we need when designing vision models. By adopting MetaFormer, it is guaranteed that the derived models would have the potential to achieve reasonable performance.

### Ablation Studies
**Replacing the pooling operator with the identity mapping:**
- Still performs well, supporting claim that MetaFormer is what is needed (vs. a particular token mixer)

**Replacing pooling with random weight matrices:**
- Still performs well (better than identity operator).

**Replace pooling with [[Depthwise Separable Kernels#Depthwise Convolution]]**
- Performs better than PoolFormer due to its better local spatial modeling ability.

**Activations**
- GELU and SiLU perform better than ReLU

**Residual connection and channel MLPs**:
- Without residual connections or channel MLP, the model can't converge.
- Shows these parts of the architecture are very important.

**Hybrid stages:**
- Pooling can handle longer input sequences (linear complexity) while attention and spatial MLP are good at capturing global information.
- You can stack MetaFormers with pooling in the bottom layers and attention or spatial FC in the top stages.
- The hybrid model performed well. Results indicate that combining pooling with other token mixers for MetaFormer may be a promising direction to further improve the performance.


# Related Work
- "Attention is not all you need: Pure attention loses rank doubly exponentially with depth" (Dong et al.) proves that without residual connections or MLPs, the output converges doubly exponentially to a rank one matrix.
- "Do vision transformers see like convolutional neural networks?" compares the feature difference between ViT and CNNs, finding that self-attention allows early gathering of global information while residual connections greatly propagate features from lower layers to higher ones.
- "How do vision transformers work?" shows that multi-head self-attentions improve accuracy and generalization by flattening the loss landscapes.