# Recurrent Convolutional Network

Tags: EECS 498, Model

### Motivation:

With a CNN+LSTM architecture:

- Inside CNN: each value is a function of a fixed temporal window (local temporal structure).
- Inside RNN: each vector is a function of all previous vectors (global temporal structure).

We want an architecture that can merge both approaches, so we can take advantage of local and global temporal structure at the same time.

### Multi-layer RNN

We can use a similar structure to the multi-layer RNN to try to improve our model. 

In our multi-layer RNN, we had a 2D grid. At each point in the grid, the output depended:

- The output vector in the previous layer at the same time step
- The output vector from the same layer at the previous time step.

You can process sequences of any length by sharing weights.

![[AI-Notes/Video/recurrent-convolutional-network-srcs/Screen_Shot.png]]

# Recurrent CNN

We can do a similar thing with a multi-layered recurrent convolutional network.

![[AI-Notes/Video/recurrent-convolutional-network-srcs/Screen_Shot 1.png]]

Now, at each point in this grid, you convolve over the features from the previous layer at the same time step, as well as the features from the same layer at the previous time step. 

- The entire network uses 2D features maps: $C \times H \times W$
- Use different weights at each layer
- Share weights across time

Now, layer 1 will extract local temporal information, but the recurrence relation allows this to be fused with information from previous time-steps, which will allow for a better fusion of local and global temporal information.

### Vanilla Recurrent CNN

Our vanilla RNN was formulated as $h_{t+ 1} = \tanh(W_hh_t + W_xx)$. We can just take this and replace all the matrix multiplies with 2D convolutions.

![[AI-Notes/Video/recurrent-convolutional-network-srcs/Screen_Shot 2.png]]

### Problem

Although this idea sounds nice, it isn't often used in practice. This is because RNNs are slow for long sequences. You can't parallelize the operations, since you need the outputs from the previous time-step.