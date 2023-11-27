---
tags: [flashcards]
source: https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html
summary:
---

Adaptive average pooling will ensure you always get the same `output_size` after applying the pooling. Example:
```python
# target output size of 5x7
m = nn.AdaptiveAvgPool2d((5, 7))
input = torch.randn(1, 64, 8, 9)
output = m(input)
# target output size of 7x7 (square)
m = nn.AdaptiveAvgPool2d(7)
input = torch.randn(1, 64, 10, 9)
output = m(input)
# target output size of 10x7
m = nn.AdaptiveAvgPool2d((None, 7))
input = torch.randn(1, 64, 10, 9)
output = m(input)
```

In average-pooling or max-pooling, you essentially set the stride and kernel-size by your own, setting them as hyper-parameters. You will have to re-configure them if you happen to change your input size.

In Adaptive Pooling on the other hand, we specify the output size instead. And the stride and kernel-size are automatically selected to adapt to the needs. 

**Output Dimensions of a CONV layer**: $N \times C_{\text{out}} \times H' \times W'$

- $H'$: $\lfloor (H-K + 2P)/S + 1 \rfloor$
- $W': \lfloor (W - K + 2P)/S + 1 \rfloor$
- Kernel size (filter size): $K_H \times K_w$
- Number filters: $C_{\text{out}}$
- Padding: $P$.
    - SAME padding preserves the original dimension
        - $P = (K-1)/2$ floored
- Stride: $S$

It matters a lot whether the input dimension (`Cin`) is an integer multiple of the output dimension (`Cout`). The flooring that occurs when calculating the output dimension $H', W'$ will cause issues otherwise.

When they are multiples of each other, then the adaptive layer's kernels are equally-sized and non-overlapping, and are exactly what would be produced by defining kernels and a stride based on the following rule:

The following equations are used to calculate the value in the source code.

```
Stride = (input_size//output_size)  
Kernel size = input_size - (output_size-1)*stride  
Padding = 0
```

[[Source]]

However, if the input dimension is a not a multiple of the output dimension, then PyTorch's adaptive pooling rule produces kernels which overlap and are of *variable size*.

Since the non-adaptive pooling API does not allow for variably-sized kernels, in this case it seems to me *there is no way to reproduce the effect of adaptive pooling by feeding suitable values into a non-adaptive pooling layer*.