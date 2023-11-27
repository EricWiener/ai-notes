---
tags: [flashcards]
aliases: [BatchNorm, BN, batch norm]
source: [[Batch Normalization.pdf]]
summary: per-dimension normalization using the mean and variance calculated from each mini-batch (per-dimension).
---

This is a great tutorial to understand the computation graph: [https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)

### Use of $\lambda, \beta$ terms:
In BatchNorm, you do a ==per-dimension== normalization using information from each mini-batch:
$$
\widehat{x}^{(k)}=\frac{x^{(k)}-\mathrm{E}\left[x^{(k)}\right]}{\sqrt{\operatorname{Var}\left[x^{(k)}\right]}}
$$
<!--SR:!2025-05-26,715,310-->

This sets each dimension in your input to have zero-mean and unit variance. In order to maintain representational ability (in case zero mean and unit variance isn't the best distribution for the model to use), BatchNorm adds learnable parameters $\lambda$ and $\beta$ and then uses $y^{(k)}=\gamma^{(k)} \widehat{x}^{(k)}+\beta^{(k)}$.

By setting $\gamma^{(k)}=\sqrt{\operatorname{Var}\left[x^{(k)}\right]}$ and $\beta^{(k)}=\mathrm{E}\left[x^{(k)}\right]$ you can recover the original activations.

> [!note]
> This is done to make sure that representational power is not lost and that the layer could learn to represent the identity function (if that were the optimal thing to do).
> 

$y^{(k)}=\gamma^{(k)} \widehat{x}^{(k)}+\beta^{(k)}$ is referred to as an **affine transform** which is “any transformation that preserves collinearity (i.e., all points lying on a line initially still lie on a line after applying the transformation)”. This just means it is a linear operation.

### You don't need a bias term for preceeding layer when using BatchNorm
Batchnormalization already includes the addition of the bias term. Recap that BatchNorm is already:
```python
gamma * normalized(x) + bias
```
So there is no need (and it makes no sense) to add another bias term in the convolution layer. Simply speaking BatchNorm shifts the activation by their mean values. Hence, any constant will be canceled out:

This shows that adding a constant to your input and then subtracting the mean of that input (as batch norm does) results in the same values:
```python
x = torch.randn(7)
x

def batchnorm_components(value):
    print("mean:", torch.mean(value))
    print("variance:", torch.var(value))
    print("value - mean:", value - torch.mean(value))
```

```
>>> batchnorm_components(x)
mean: tensor(0.5277)
variance: tensor(1.4118)
value - mean: tensor([-1.1636,  2.2207, -0.3310, -0.6481, -1.0293,  0.7440,  0.2074])

>>> batchnorm_components(x + 10)
mean: tensor(10.5277)
variance: tensor(1.4118)
value - mean: tensor([-1.1636,  2.2207, -0.3310, -0.6481, -1.0293,  0.7440,  0.2074])
```
As you can see you end up with the same values after subtracting the mean regardless of whether you add a constant or not. A bias term in a conv/linear layer will just add a constant to a certain channel and batch norm will subtract the mean per-channel across a batch. [My SO Post](https://stackoverflow.com/a/76191563/6942666)

> [!NOTE] Alternate explanation
> Batch normalization eliminates the need for a bias vector in neural networks, since it introduces a shift parameter that functions similarly as a bias. The shift term in batch normalization is also a vector with the same number of channels as the original bias would have, for instance the documentation for `BatchNorm2d` in Pytorch reads: "The mean and standard-deviation are calculated per-dimension over the mini-batches.

Note that although you don't need a bias, this doesn't mean you will get the same result when using a bias/not using a bias:
```python
import torch
import torch.nn as nn

torch.manual_seed(0)
module_w_bias = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
    nn.BatchNorm2d(num_features=16)
).eval()

torch.manual_seed(0)
module_no_bias = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(num_features=16)
).eval()

x = torch.randn(1, 3, 10, 10)

print(torch.allclose(module_w_bias(x), module_no_bias(x))) # False
```

In the above example, I'm setting the random seed to 0 for both the `module_w_bias` and `module_no_bias` modules. This means that the initial parameters of the modules are the same for both cases, which is why you might expect them to produce the same output.

However, when you use batch normalization in your model, the statistics of the batch normalization layers (mean and variance) depend on the input to the layer. In other words, batch normalization adjusts the scale and shift of the data based on the mean and variance of the data in the current batch.

So even if the weights of the convolutional layer are the same in both `module_w_bias` and `module_no_bias`, the output of the batch normalization layer can be different because the input to the layer is different. This difference in the output can cause a slight difference in the final prediction of the model.

Therefore, it's not necessary to include a bias term in the `nn.Conv2d` layer