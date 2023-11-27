---
tags:
  - flashcards
source: https://youtu.be/2V3Uduw1zwQ
aliases:
  - layer normalization
summary: normalization typically used by transformers
---
![[Screenshot_2022-02-05_at_08.28.582x.png]]
[Source](https://arxiv.org/abs/1803.08494)

Input values in all neurons in the same layer are normalized for each data sample so all neurons in the same layer will have the same mean and the same variance.

Here is a good comparison between Batch Norm and Layer Norm:
![[Layer Normalization Deep Learning Fundamentals.mp4]]
### Reasons to use instead of [[Batch Normalization]]
**Batch Normalization Cons:**
Batch Normalization is hard to use with sequence data (ex. RNNs) because if the sequences are of different lengths, batch norm is hard to calculate. 

It is also hard to use with small batch sizes since the a smaller batch will result in a poor estimate of the mean and std dev of the whole dataset.

It is also hard to parallelize a network that uses batch norm since you would need to sync the mean and std dev calculations from each replica of the model.

**LayerNorm Pros:**
LayerNorm calculates the mean and standard deviation across all channels for a single training example. This makes it so it has no dependency on the batch size and is easy to parallelize across multiple replicas of the model. It also means the model has the same behavior at test time and training time.

LayerNorm is better for RNNs since the batch doesn't matter and you can better handle sequences since you just need to look at the token being currently generated.

**LayerNorm Cons:**
It doesn't always work well with CNN networks.
# Where to apply LayerNorm?
![[layernorm-20220904092429354.png]]
#### Post-norm
In [[Attention is All You Need]] the normalization was initially applied after element-wise residual addition (post-norm):
$$x_{l+1}=\mathrm{LN}\left(x_l+\mathcal{F}\left(x_l ; \theta_l\right)\right)$$
```python
class MultiHeadAttention(nn.Module):
    def forward(self, q, k, v, mask=None):
        # ...
        q += residual
        q = self.layer_norm(q)
        return q, attn
```
[Source](https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/132907dd272e2cc92e3c10e6c4e783a87ff8893d/transformer/SubLayers.py#L9)

The issue with this is that when backpropagating all the gradients have to flow back through the LayerNorm and there isn't a direct path through the residual paths.

### Pre-norm
However, later papers [[Learning Deep Transformer Models for Machine Translation]] (and earlier papers) suggested that the pre-norm was better than post-norm. In pre-norm, normalization is applied to the input to every sublayer.
$$x_{l+1}=x_l+\mathcal{F}\left(\mathrm{LN}\left(x_l\right) ; \theta_l\right)$$
In [[ViT An Image is Worth 16x16 Words Transformers for Image Recognition at Scale|ViT]], LayerNorm is applied before every block, and residual connections after every block. ViT does this based on the work in [[Learning Deep Transformer Models for Machine Translation]]. See [[ViT An Image is Worth 16x16 Words Transformers for Image Recognition at Scale#ViT Transformer Code|ViT code]].

![[Learning Deep Transformer Models for Machine Translation#Pre-norm over post-norm]]

### Residual post norm
![[Swin v2#Residual post norm]]

