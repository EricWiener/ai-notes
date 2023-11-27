---
summary: Most popular activation function. No saturation in positive region. Still isn’t zero-centered.
url: "[cs231n](https://cs231n.github.io/neural-networks-1/#actfun)"
---
![[relu-graph.png.png]]
ReLU just computes `out = max(0, x)`. All you have to do is check the signed bit of a floating point number. If it is positive we leave it alone, otherwise we set it to zero. This makes it very computationally efficient.

Unlike `tanh` and `sigmoid`, it does not saturate (go flat) in the positive region. It converges much faster than either in practice (approximately 6x as fast).

However, it still isn't zero centered, so it still has the issues of gradients all being positive or negative. However, in practice it still converges, so it isn't that bad a problem.

AlexNet was one of the first papers to demonstrate the usefulness of ReLUs.

### Dead ReLU
[Blog post with a good explanation](https://towardsdatascience.com/the-dying-relu-problem-clearly-explained-42d0c54e0d24#4995)

Unfortunately, ReLU units can be fragile during training and can “die”. For example, a large gradient flowing through a ReLU neuron could cause the weights to update in such a way that the neuron will never activate on any datapoint again. 

![[zero-region-of-relu.png]]
If you have a layer where the pre-activation output is all negative, after passing it through the ReLU, the output will be all zeros (since negative values get set to exactly 0 - see the red region circled above). This will then cause the local gradient to become zero and no updates will occur. The ReLU units will die and they can't be brought back since the pre-activation output remains negative and the gradient remains zero.

This is sometimes avoided by initializing ReLU neurons (layers with ReLU activations) with slightly positive biases (e.g. 0.01). This can also be helped by using BatchNorm.

### Example: Approximation $f: R \rightarrow R$

We can think of a neural network with just one layer + ReLU as combining four scaled and shifted ReLU's. 

![[relu-approximation.png.png]]

$y = u_1 \cdot h_1 + u_2 \cdot h_2 + u_3 \cdot h_3 + p$ is just the sum of scaled ReLU functions because all $h_1, h_2, h_3$ are all just ReLU functions.

The final output of a ReLU network is a piece-wise linear function (each ReLU is piece-wise linear and we just combine them). **The number of linear regions scales linearly if you have a single layer. If you have more layers, the number of linear regions you can learn scales exponentially with the depth of the network.** You can use very complex functions with many linear pieces using not that many pieces in each layer. 

In a single layer, the complexity of the function you learn grows linearly. In a deep neural network, the complexity of the function grows exponentially.

You can do similar things with other activation functions as with ReLU (example sigmoid).

![[relu-bumps.png.png]]

## Forward:

```python
def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).
  Input:
  - x: Input; a tensor of any shape
  Returns a tuple of:
  - out: Output, a tensor of the same shape as x
  - cache: x
  """
  out = None

  out = x.clone()
  out[out < 0] = 0
  
  cache = x
  return out, cache
```

## Backward:

```python
def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).
  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout
  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
	
	# Avoid modifying the upstream gradient
  dx = dout.clone()
	# Set all elements of the upstream gradient where the input had values
	# less than zero -> 0
  dx[x <= 0] = 0
 
  return dx
```

```python
Started first linear + ReLU
x.shape torch.Size([100, 3, 32, 32])
w.shape torch.Size([3072, 100])
b.shape torch.Size([3072])
```