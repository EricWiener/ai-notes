---
tags: [flashcards]
source:
summary:
aliases: [mlp, feedforward layer, multi-layer perceptron, perceptron, FFN, MLP]
---
![[linear-layer-diagram.png]]
Make note that $b_j$ is not included in the summation term

# Forward

The linear layer computes $f(x) = Wx + b$ where $x$ is the input of size $(N, D)$ and $W$ is weights of size $(D, M)$. Every dimension of $x$ gets a corresponding row of weights in $W$. The number of columns of $W$ correspond with how many rows the output of the layer will have. $b$ is added on after $Wx$ is computed, so it will be of size $(M,)$ so it can be broadcast onto the output of $Wx$ which is $(N, M)$.

```python

  def forward(x, w, b):
	  """
	  Computes the forward pass for an linear (fully-connected) layer.
	  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
	  examples, where each example x[i] has shape (d_1, ..., d_k). We will
	  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
	  then transform it to an output vector of dimension M.
	  Inputs:
	  - x: A tensor containing input data, of shape (N, d_1, ..., d_k)
	  - w: A tensor of weights, of shape (D, M)
	  - b: A tensor of biases, of shape (M,)
	  Returns a tuple of:
	  - out: output, of shape (N, M)
	  - cache: (x, w, b)
	  """
	  out = None
	  #############################################################################
	  # TODO: Implement the linear forward pass. Store the result in out. You     #
	  # will need to reshape the input into rows.                                 #
	  #############################################################################
	  # Replace "pass" statement with your code
	  # (N, D) <= (N, d_1, ..., d_k)
	  x_flat = x.flatten(start_dim=1)
	  # (N, M) = (N, D) x (D, M)
	  out = x_flat.mm(w)
	  # (N, M) = (N, M) + (M,)
	  out += b.reshape(1, -1)
	  #############################################################################
	  #                              END OF YOUR CODE                             #
	  #############################################################################
	  cache = (x, w, b)
	  return out, cache
```

# Backward

The backward pass needs to calculate $\frac{\partial W}{\partial L}$ (the change in the weights with respect to the total Loss). You also need to calculate $\frac{\partial b}{\partial L}$ if it is not included in $W$ with the bias trick. Finally, you need to compute $\frac{\partial x}{\partial L}$ so it can be used as the upstream gradient for the preceding layer since $x$ is really the output of the previous layer.

```python
def backward(dout, cache):
  """
  Computes the backward pass for an linear layer.
  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)
    - b: Biases, of shape (M,)
  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the linear backward pass.                                 #
  #############################################################################
  # Replace "pass" statement with your code
  # (N, D) = (N, M) x (D, M).T = (N, M) x (M, D) = (N, D)
  dx = dout.mm(w.t())
  # (N, d1, ..., d_k) <= (N, D)
  dx = dx.reshape(x.shape)

  x_flat = x.flatten(start_dim=1)
  # (D, M) = (N, D).T  x (N, M) = (D, N) x (N, M)
  dw = x_flat.t().mm(dout)

  # (M,) <= (N, M)
  db = dout.sum(dim=0)
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################
  return dx, dw, db
```

We can derive the update rules for `dx, db, dw`. First we can write out the expanded form of the matrix multiplication (b is broadcasted to make its shape clearer).
![[IMG_8755AC158D74-1.jpeg.jpeg]]
We can then draw out the computation graph. Note that the chain rule is commutative in 1D, but it is not with matrices. You need to see which way the shapes work out correctly.
![[IMG_F32765A10F39-1.jpeg.jpeg]]
We can then solve to find the gradient of the loss with respect to $b$. This explains why we do `db = dout.sum(dim=0)` in PyTorch. Note this explanation could have been simplified just by looking at the computation graph since we assign a place holder variable $p = Wx$. If we take the gradient of $z = p + b$, we can clearly see that it is just a vector of ones since the $p$ goes to 0.
![[IMG_4A2328708B99-1.jpeg.jpeg]]
Addition in a computation graph simply copies the upstream gradient to both incoming paths, so we can now just solve for