---
tags: [flashcards]
source: [[Group Normalization.pdf]]
summary: Introduces Group Normalization. It does better than BN for small batch sizes, but worse for larger batch sizes.
aliases: [Groupnorm, groupnorm, group norm]
---

> [!note]
> GroupNorm showed improvements for batch sizes of 8, 4, and 2, but BatchNorm did better for batch sizes of 32 and 16.
> 

- Normalizing along the batch dimension introduces problems — BN’s error increases rapidly when the batch size becomes smaller, caused by inaccurate batch statistics estimation.
- GN divides the channels into groups and computes ==within each group== the mean and variance for normalization. GN’s computation is ==independent== of batch sizes.
- GN is much better than BN for smaller batch sizes and comparable/slightly worse for larger batch sizes.
- GN does not exploit the batch dimension, and its computation is independent of batch sizes.
- LN and IN have limited success in visual recognition, for which GN presents better results.
- The concept of “batch” is not always present, or it may change from time to time. BN is not legitimate at inference time, so the mean and variance are pre-computed from the training set [26], often by running average
- The higher-level layers are more abstract and their behaviors are not as intuitive (as the first conv layer) when trying to consider the reasoning why grouping channels makes sense.
- BN, LN, IN, and GN normalize with $\hat{x}_{i}=\frac{1}{\sigma_{i}}\left(x_{i}-\mu_{i}\right) .$
- GN becomes LN if we set the group number as G = 1.
- GN becomes IN if we set the group number as G = C (*i.e*., one channel per
group).
<!--SR:!2027-11-21,1624,350!2027-12-01,1634,352-->


![[Screenshot_2022-02-05_at_08.28.582x.png]]
Note that the spatial dimensions (height and width) are shown on a single axis to make the diagram clearer.

[GIST explaining different types of normalization](https://gist.github.com/radekosmulski/b708e2367fe78ee21ffba382633e52d3#summary-of-fastaiais-in-depth-discussion-of-types-of-normalization)

### [[Batch Normalization|BatchNorm]]
- BN normalizes the features by the mean and variance computed within a (mini-)batch.
- It is required for BN to work with a *sufficiently large batch size* (*e.g*., 32 per worker
- Reducing BN’s batch size increases the model error dramatically

### GroupNorm Implementation
![[Screenshot_2022-02-05_at_08.45.232x.png]]

```python
class GroupNorm(Module):
    r"""Applies Group Normalization over a mini-batch of inputs

    Args:
        num_groups (int): number of groups to separate the channels into
        num_channels (int): number of channels expected in input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to ``True``, this module
            has learnable per-channel affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, C, *)` where :math:`C=\text{num\_channels}`
        - Output: :math:`(N, C, *)` (same shape as input)
    """
    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine']
    num_groups: int
    num_channels: int
    eps: float
    affine: bool

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True,
                 device=None, dtype=None) -> None:
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.empty(num_channels, **factory_kwargs))
            self.bias = Parameter(torch.empty(num_channels, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.group_norm(
            input, self.num_groups, self.weight, self.bias, self.eps)
```

- Specifically, the pixels in the same group are normalized together by the same μ and σ. GN also learns the per-channel $\lambda$ and $\beta$ (these are used if `affine=True` in the torch implementation)
- The $\lambda$ and $\beta$ are learned per-channel instead of per-group. This is done by all normalization layers to make sure that representational power is not lost and that the layer could learn to represent the identity function (if that were the optimal thing to do). [[See here]]
    
    

```python
def GroupNorm(x, gamma, beta, G, eps=1e−5):
	# x: input features with shape [N,C,H,W]
	# gamma, beta: scale and offset, with shape [1,C,1,1] # G: number of groups for GN
	N, C, H, W = x.shape
	x = tf.reshape(x, [N, G, C // G, H, W])

	mean, var = tf.nn.moments(x, [2, 3, 4], keep dims=True)
	x = (x − mean) / tf.sqrt(var + eps)

	x = tf.reshape(x, [N, C, H, W])

	return x ∗ gamma + beta
```

### Results

- For a batch size of 32, the slightly higher validation error of GN implies that GN loses some regularization ability of BN.
- BN benefits from the stochasticity under some situations, its error increases when the batch size becomes smaller and the uncertainty gets bigger.
- GN’s behavior is more stable and insensitive to the batch size.
- BN has been so influential that many state-of- the-art systems and their hyper-parameters have been de- signed for it, which may not be optimal for GN-based mod- els.

### Mentioned Work

- Using **asynchronous solvers** (ASGD [10]), a practical solution to large-scale training widely used in industry is not possible if you sync BN across multiple GPUs.
- **ResNeXt** [63] investigates the trade-off between depth, width, and groups, and it suggests that a larger number of groups can improve accuracy under similar computational cost.
- **MobileNet** [23] and **Xception** [7] exploit *channel-wise* (also called “depth-wise”) convolutions
- **ShuffleNet** [65] proposes a channel shuffle oper- ation that permutes the axes of grouped features.