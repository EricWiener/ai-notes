This is a great explanation of the backward pass: [https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c](https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c)

This is also a good walkthrough of the CONV backprop code: [https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509](https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509) 

![[1_CkzOyjui3ymVqF54BR6AOQ.gif.gif]]

![[AI-Notes/Layers/2d-convolution-implementation-srcs/Untitled.png]]


Simple code:

[https://github.com/CatalinVoss/cnn-assignments/blob/master/assignment2/cs231n/layers.py](https://github.com/CatalinVoss/cnn-assignments/blob/master/assignment2/cs231n/layers.py)

# Backward

```python
def backward(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  # Unpack the cached values
  x, w, b, conv_param = cache

  # Pad the tensor (without modifying original)
  pad = conv_param['pad']
  padded_x = torch.nn.functional.pad(x, (pad, pad, pad, pad), "constant", 0)

  # Create empty gradients filled with zeros
  padded_dx = torch.zeros(padded_x.shape, dtype=x.dtype, device=x.device)
  dw = torch.zeros(w.shape, dtype=w.dtype, device=w.device)
  db = torch.zeros(b.shape, dtype=b.dtype, device=b.device)

  # Get dimensions
  N, C, H, W = x.shape # N, C, H, W
  F, _, HH, WW = w.shape # F, C, HH, WW
  padded_H = H + 2 * pad
  padded_W = W + 2 * pad
  stride = conv_param['stride']

  _, _, h_prime, w_prime = dout.shape # N, F, H', W'

  # convolve with the filter
  out_r, out_c = 0, 0
  for n in range(N): # nth training example
    for r in range(0, padded_H-HH+1, stride): # start row of x area
      for col in range(0, padded_W-WW+1, stride): # start col of x area
        for f in range(F): # fth filter
          row_start, row_end = r, r + HH
          col_start, col_end = col, col + WW

          # compute db. This is irrespective of the channel
          db[f] += dout[n][f][out_r][out_c]

          for chan in range(C): # loop over the channels
            x_region = padded_x[n, chan, row_start:row_end, col_start:col_end]

            # compute dw for the fth filter
            dw[f][chan] += x_region * dout[n][f][out_r][out_c]

            # compute dx
            padded_dx[n, chan, row_start:row_end, col_start:col_end] += w[f][chan] * dout[n][f][out_r][out_c]
        # End loop through filters
        out_c += 1
      # End loop through columns
      out_r += 1
      out_c = 0
    # end loop through rows
    out_r = 0 # reset number of rows
  # end loop through N

  # Remove padding from padded_dx
  dx = padded_dx[:, :, pad:pad+H, pad:pad+W]
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################
  return dx, dw, db
```