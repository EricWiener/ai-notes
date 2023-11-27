---
tags: [flashcards, eecs498-dl4cv]
aliases: [RNN, LSTM, Captioning]
url: https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
---
Recurrent Neural Networks allow us to work with sequences of both input and output data. Previously, we have been working with just one input and one output (one image ⇒ one label). This is an example of a **feed-forward neural network**, where you just pass the data along between layers. 

Recurrent Neural Networks allow you to pass data between layers, receive more than one input, and output more than one output.

### Motivation for handling sequences of data: handle more types of tasks
![[types-of-tasks.png|Types of tasks]]
- One to one: this is a "Feedforward" Neural Network (e.g. image classification: image → label)
- One to many: e.g. image captioning (image → sequence of words)
- Many to one: video classification (sequence of images → label)
- Machine translation (sequence of words → sequence of words in different language)
- Many to many: per-frame video classification (sequence of images → sequence of labels). This could be something like saying the first ten frames are someone dribbling a basketball, next three are shooting the basketball, and last five are celebrating.

Being able to handle sequences of data opens up a variety of new tasks that we can handle.

# How it works

The RNN has an "internal state" that is updated as the sequence is processed. Each time it receives an input, it will give an output and also update its state. It can be represented as a **recurrence formula** at every time step:

$$
h_t = f_W(h_{t-1}, x_t)
$$

- $h_t$ is the new state of the node
- $f_W$ is some function with parameters $W$
- $h_{t-1}$ is the old state
- $x_t$ is the input vector at some time step

### Parameter Sharing

The function $f_W$ uses the same $W$ for every time step. This uses parameter sharing. If you didn’t use parameter sharing, then every possible sequence length would have a different set of weights. This would be a huge number of parameters and the system doesn’t generalize.

- Each member of the output is a:
    - Function of the previous members of the output
    - Produced by the same update rule applied to previous outputs
- We can process any sequence length with the same function $f_W$.

### Vanilla Recurrent Neural Network (aka Elman RNN)

This is a very simple recurrent neural network.  It isn't used in modern networks, but it important for historical reasons.

- The state consists of a single "hidden" vector **h**
- $h_t = f_W(h_{t-1}, x_t)$
- $h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b)$
    - $W_{hh}$ is the weight matrix that is applied to the previous hidden state
    - $W_{xh}$ is applied to the input at the current time step
    - $b$ the learnable bias term
    - Uses $\tanh$ non-linearity. It is preferable to sigmoid because it is zero-centered. We don't use ReLU because an RNN has so many time-steps and if you **used ReLU, the activations could just keep growing, causing overflow**. Tanh isn't perfect, but it squishes the outputs, avoiding overflow. Tanh isn't used much in modern networks.
- $y_t = W_{hy}h_t$
    - $W_{hy}$ is a third weight matrix that is applied to the final hidden state. This is just a linear transformation.

![[rnn-computational-graph.png|Example of a many to one computational graph.]]

The above is an example of a **many-to-one** RNN. $y$ would be the output of $W_{hy}h_T$ at the final time step. You can also output $y_t$ for multiple hidden states if you want to have multiple outputs ($y_t = W_{hy}h_{t}$).

Because you use the same weight matrix at every time, if you unroll the computation graph, you will see the weight matrix $W$ has many connections throughout the network. You will need to sum the gradients calculated from all these connections to calculate the total gradient.

![[rnn-computational-graph-many-to-many.png|You now have multiple outputs and multiple inputs.]]

This is similar to an HMM ([[Hidden Markov Model]]) in that each output can be computed just using the previous hidden state (although this hidden state is affected by the entire graph and therefore $y_t$ implicitly contains information from the entire graph).

If you also have outputs at each step, you can calculate the loss at each step. You will then need to add all these losses up.

#### RNN Backward Pass Code
```python
def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H). 
    
    NOTE: 'dh' contains the upstream gradients produced by the 
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    N, T, H = dh.shape

    dx = torch.zeros((N, T, D), device=dh.device, dtype=dh.dtype)
    dh0 = torch.zeros((N, H), device=dh.device, dtype=dh.dtype)
    dWx = torch.zeros((D, H), device=dh.device, dtype=dh.dtype)
    dWh = torch.zeros((H, H), device=dh.device, dtype=dh.dtype)
    db = torch.zeros((H,), device=dh.device, dtype=dh.dtype)
    
    dprev_h = torch.zeros_like(dh[:, 0, :])

    for i in range(T - 1, -1, -1):
      # When calculating the gradient for the hidden state, it receives both a gradient
      # from the loss calculated at its specific time interval, as well as the upstream
      # gradient. Because the node has two arrows coming back to it during back-prop,
      # you need to add them together. dh[:, i, :] is the gradient with respect to
      # the loss function for this time-step and dprev_h is the upstream gradient
      _dx, dprev_h, _dWx, _dWh, _db = rnn_step_backward(dh[:, i, :] + dprev_h, cache[i])
      dx[:, i, :] = _dx # update the gradient for each time-step input
      dWx += _dWx
      dWh += _dWh
      db += _db
    
    # update the first hidden state gradient
    dh0 = dprev_h

  return dx, dh0, dWx, dWh, db
```

When calculating the gradient for the hidden state, it receives both a gradient from the loss calculated at its specific time interval, as well as the upstream gradient. Because the node has two arrows coming back to it during back-prop, you need to add them together.



## Sequence to Sequence (seq2seq)

![[seq-2-seq.png]]

If you want to take in a sequence (English words) and output another sequence (French words), the sequences don't always have the same length. You can combine a many-to-one with a one-to-many network to handle this. Note that you often use different weight matrices for the encoder and decoder ($w_1$ and $w_2$ here respectively).

![[many-to-one.png|The many to one  network encodes the input sequence into a single vector (the last hidden state generated by the network).]]

The **many to one** network encodes the input sequence into a single vector (the last hidden state generated by the network). This allows the network to handle a variably sized input sequence, but always produce the same size final hidden state to feed to the decoder.

![[one-to-many.png|The one to many network decodes the hidden state into an output sequence.]]

This hidden state is then given to the **one to many** network which produces an output sequence from the single input vector. 

## Example: Language Model
In this example we want to predict the next character given the current character and the previous inputs. You can encode a training sequence "hello" into a one-hot encoding for the characters [h, e, l, o]. You can then learn a model that will give predictions for the next character. The inputs to the model (red) are one-hot encoded vectors.

![[language-model.png|For instance, given input h and e the next predicted character is l.]]

This model's hidden state is given by $h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t)$. 

![[language-model-w-softmax.png]]

You can even generate new text with the same style as the training set if you use the output of one iteration as the input to the next iteration. In the above example, "e" is the output of the first step and is then used as the input for the second step.

### Embedding Layer

Because our inputs are just one-hot vectors, when we perform the matrix multiply $W_{xh}x_t$ we are effectively just copying over a single column of the matrix $W_{xh}$

![[embedding-layer-weights.png]]
In the above example, `[[1], [0], [0], [0]]` is the embedding for the character `h`. The encoding is a one-hot vector of shape `D x 1`. The weight matrix will be of shape `N x D` (in this case 3x4) so when you multiple `N x D` by `D x 1` you get output `N x 1`.

We can make this more efficient by introducing an **emedding layer.** This goes between the input of the network and the hidden layer. This embedding layer just picks out the corresponding column from the weight matrix (which performs a one-hot sparse matrix multiply implicitly). Each embedding layer corresponds to a single element in the vocabulary. It recieves a one-hot input and gives a dense vector corresponding to the column of the weight matrix (basically performing a look-up for us).
![[language-model-w-embedding-layer.png]]

#### What's the point of an embedding layer in an LSTM
??
An embedding layer is a computation trick to convert the one-hot encoded input to its corresponding column in the weight matrix.
![[language-model-w-embedding-layer.png]]
<!--SR:!2025-10-26,999,310-->

## Backpropagation through time

You first go forward through the entire sequence to compute loss, and then go backward through the entire sequence to compute the gradient. 

![[backprop-through-time.png]]

However, this takes a lot of memory for long sequences since you have to store all your intermediate values for back-propagation. So it isn't ideal for long sequences. This is also slow if your sequence is very long because you have to wait for the entire forward pass to complete before you could make any gradient update.

## Truncated backpropagation through time

![[trunacated-backprop-through-time.png]]
To avoid having to store the whole history of the sequence, you can run forward and backward passes through chunks of the sequence instead of the whole sequence. The gradients computed are just an approximation, but they are hopefully close enough for learning. 

- You carry hidden states forward in time forever, but only backpropogate for some smaller number of steps.
- You go forward through one chunk, and hold onto the output
- You then backpropogate through this chunk and compute weight updates
- You then move onto the next chunk and pass in the hidden state you held onto as input and use the updated weights you just calculated.
- You repeat the process of forward pass (and store final hidden state), backward pass, and update weights for each chunk.

The forward pass is still processing an infinite sequence, but the backward pass is much smaller, so you don't have to keep as much data in GPU memory.

Note that this makes sense for one-to-many or many-to-many tasks where you compute losses at multiple points, but it doesn't make sense if you have one-to-one or many-to-one since you don't have intermediate losses.

# Image Captioning

![[image-captioning.png]]

You can use a CNN to encode the features of an image. You can then pass the extracted features as the input to an RNN. The RNN can then generate words based on the input. The sentence ends with the END output.

You can train this on a database of images and their captions.

![[transfer-learning-image-captioning.png]]

In the above example, you can take a CNN trained on ImageNet and remove the fully connected layer. We then have a feature representation of the image. We modify our function to calculate the next hidden state from

- $h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b)$
- To: $h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + W_{ih}v + b)$

The new feature $v$ is the encoding of the image learned by the CNN. $W_{ih}$ is the weight matrix used for this encoding. Now, when computing the new hidden state, you use the output of the previous hidden-state, the input at this time step, and the feature encoding of the image (this gets passed to each level).

When training, you can use a technique called **[[Teacher Forcing]]** that will always provide the previous ground truth word (instead of the previous predicted output). This makes training easier and reduces the need to model long-range dependencies.

**Starting and stopping sequence**

In order for the model to give labels of finite length, when training you can start labels with `<start>` and end them with `<end>`. This way to start the label generation, you can just use `<start>`, so you don't need to specify the first real world. The sequence then stops when `<end>` is outputted.

> [!note]
> Even a simple nearest neighbor approach can get high accuracy on image captioning (provide the label of the most similar image).

**Training with batches of data**
It's hard to train with batches of data if you were doing something like image captioning because each caption could have a different length. It's hard to do this in native PyTorch, but Nvidia provides a RNN library to assist with this that implemented in CUDA. It will unroll the different elements in the batch for different numbers of timesteps.

In PyTorch you would often pad the ending of the caption with `NULL` tokens so all captions were the same length and then not have elements of your sequence that are `NULL` tokens not contribute to the calculated loss.

# Vanilla RNN Gradient Flow
![[gradient-flow.png]]
Note the `stack` refers to concatenating the hidden state and the input to be multiplied by a single $W$ matrix (where $W$ contains both the weights applied to the hidden state and the weights applied to $x_t$).

**What is the problem with RNNs?**
?
When you back propagate through the Vanilla RNN from $h_t$ to $h_{t-1}$ you need to first pass $h_t$ through the [[Tanh]] (which sends the gradient to zero at extremes), and then you need to multiply by the weight matrix $W_{hh}$.  Since you have to do backpropagate for every level, computing the gradient of $h_0$ involves many factors of $W$ and [[Tanh]].
![[multiple-gradient-flow.png]]
You can easily get **exploding gradients** (largest singular value of $W > 1$) or **vanishing gradients** (largest singular value of $W < 1$). 

- If you get exploding gradients, you can solve this with gradient clipping by scaling the gradient if its norm is too big with the following:
```python
grad_norm = torch.sum(grad * grad)
if grad_norm > threshold:
	grad *= (threshold / grad_norm) # Normalize.
```
   - Ex: if `grad_norm = 20` and `threshold = 10`, then you do `20 *= 10 / 20 -> 20 *= 1/2 = 2`. 
    - Note this is a hack - not ideal.
- If you get vanishing gradients, you need to change the RNN architecture.

# Long Short Term Memory (LSTM)

This is a way to handle the vanishing gradient problem of the Vanilla RNN. Instead of keeping a single hidden vector at each time step, we keep two: `cell state` ($c_t \in \mathbb{R}^H$) and `hidden state` ($h_t \in \mathbb{R}^H$). At every time step, you **compute four gates**.

![[rnn-and-lstm-comparison.png]]

- **i**: Input gate: whether to write to cell
- **f**: Forget gate: whether to erase cell
- **o**: Output gate: how much to reveal cell.
- **g**: Gate gate: how much to write to cell

![[lstm-gates.png]]

We originally compute $Wx$ just like we did with the Vanilla RNN: we stack $x$ and the previous hidden state $h$ and then multiply it by the weight matrix $W$. For the Vanilla RNN, this was our next hidden state. However, for the LSTM, we take this output and divide it up into four different gates. We apply different activation functions to each.

Note that $W$ is now $4h \times 2h$ instead of $h \times h$ (as it was for our Vanilla RNN).

The input, forget, and output gates all go through a [[Sigmoid]] function, so they are bounded between (0, 1).

The gate gate goes through a [[Tanh]], so it is bounded between (-1, 1).

We can then combine the different gate values to calculate our next value:

$$
c_t = f \odot c_{t-1} + i \odot g
$$

- The next `cell state` is equal to the `forget gate` * `previous cell state` + `input gate` * `gate gate`. The cell state ($c_t$) is internal to the LSTM.
- This says the new cell state is a combination of how much we forgot the previous cell state + how much we let in the input.
- $\odot$ means multiply element wise, so these operations are all applied element wise (can choose to remember or forgot element wise).

$$
h_t = o \odot \tanh(c_t)
$$

- The next hidden state is `output gate` * `tanh(cell state)`. The [[Tanh]] will squash the cell state values between -1 and 1. The output gate ($o$) is between 0 and 1 and will scale values of the squashed cell state.
- This means the output is given by a non-linear function of our current state  multiplied by how much we let out into the next hidden state. The cell-state is an internal property of the LSTM that it can choose how much to reveal via the hidden state $h_t$.

### Interpreting LSTM Cells
The following is a paper from Professor Johnson where he tried to interpet the different cells in an LSTM. Open up the PDF in Skim for the annotated version.
![[AI-Notes/Concepts/rnn-lstm-captioning-srcs/Visualizing and Understanding Recurrent Networks.pdf]]

### Forward Prop Implementation
At each time-step we first compute an **activation vector** $a\in\mathbb{R}^{4H}$ as $a=W_xx_t + W_hh_{t-1}+b$. We then divide this into four vectors $a_i,a_f,a_o,a_g\in\mathbb{R}^H$ where $a_i$ consists of the first $H$ elements of $a$, $a_f$ is the next $H$ elements of $a$, etc. We then compute the *input gate* $g\in\mathbb{R}^H$, *forget gate* $f\in\mathbb{R}^H$, *output gate* $o\in\mathbb{R}^H$ and gate *gate* $g\in\mathbb{R}^H$ as

$$
i = \sigma(a_i) \hspace{2pc}
f = \sigma(a_f) \hspace{2pc}
o = \sigma(a_o) \hspace{2pc}
g = \tanh(a_g)

$$

where $\sigma$ is the sigmoid function and $\tanh$ is the hyperbolic tangent, both applied element-wise.

Finally we compute the next cell state $c_t$ and next hidden state $h_t$ as

$$
c*{t} = f\odot c*{t-1} + i\odot g \\
h_t = o\odot\tanh(c_t)
$$

where $\odot$ is the element-wise product of vectors.

#### LSTM Forward Step Code
```python
def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b, attn=None, Wattn=None):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)
    - attn and Wattn are for Attention LSTM only, indicate the attention input and
      embedding weights for the attention input

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    """
    next_h, next_c = None, None
   
	# First we compute Activation Vector: a = W_x * x_t + W_h * h_{t-1} + b
    # (N, 4H) = (N, D) x (D, 4H) + (N, H) x (H, 4H)
    activation = x.mm(Wx) + prev_h.mm(Wh) + b

    # Split the activation vector into 4 chunks of size (N, H)
    a_i, a_f, a_o, a_g = torch.chunk(activation, 4, dim=1)

    # Apply sigmoid to i, f, and o
    i, f, o = torch.sigmoid(a_i), torch.sigmoid(a_f), torch.sigmoid(a_o)
    
    # Apply tanh to g
    g = torch.tanh(a_g)

    # Do element-wise multiplication
    # c_{t} = f * c_{t-1} + i * g
    next_c = f * prev_c + i * g
    
    # h_t = o * tanh(c_t)
    next_h = o * torch.tanh(next_c)
  
    return next_h, next_c
```

### LSTM Backprop

When going through an LSTM cell:

- You receive the previous cell state and the previous hidden state.
- You concatenate the input $x_t$ with the previous hidden state
- You multiply by the weight matrix $W$ and divide the output into four gates
- You then use the four gates to decide how to output the next cell state and the next hidden state.

![[lstm-info-flow.png|Diagram showing the flow of info in single LSTM cell.]]

When you back propagate through the LSTM from $c_t$ to $c_{t-1}$, you only need to go back through $c_t = f \odot c_{t-1} + i \odot g$. If we look at the computation graph you can see that this:

- First passes through a sum node, which just copies the gradient over. There is no information lost here.
- Then passes through the element wise matrix multiply with $F$. If you have low values in $F$, some information can be lost, but this still doesn't have to pass through a sigmoid non-linearity (pretty bad). This is similar to back-propagating through an element wise constant multiply (with respect to $c$).

Now, you don't have to go through any non-linearities or matrix multiplies. Much less information is lost. **Note, you still need to back-prop through the weights,** but at least you have one path where the gradient is preserved.

![[lstm-uninterrupted-gradient-flow.png|You now have a much better gradient flow through the RNN.]]

> [!note]
> With the LSTM, there will always be some pathway in which the gradient is retained. This will hopefully be enough to keep the flow of the gradient going.
This is a **similar concept of [[ResNet]]'s additive skip connections**. The additive connections give us uninterrupted gradient flow through time steps.

# Multilayer RNN
Although RNN's can have many hidden layers between the initial input and the final output, if you have many outputs, you don't necessarily have that many layers between your input and your output (sometimes just one layer). This is too shallow and you can use multiple layers of hidden states to learn more complex functions.

![[multi-layer-rnn.png]]

You can have multiple layers of an RNN. You pass the hidden state of the first layer to the second layer as its inputs. The second layer then gives the output values. You can repeat this with multiple layers, but usually this is done with 2-3 layers (not done with as great a depth as NN).

You would usually use different weight matrices for different layers of the RNN (just like NN has different weights for different layers).

# RNN Problems
- Results in very deep networks (often slow, hard to train) 
- [[Attention#Attention Layer|Attention]] can be used sometimes as an alternative to RNNs.

# Summary
- RNNs allow a lot of flexibility in architecture design
- Vanilla RNNs are simple, but don't work well
- Common to use LSTM or [[GRU]] (not covered in depth - similar to LSTM). Additive interactions improve gradient flow.
- People have started to use neural architecture search to learn more complex RNN architectures.
- Backward flow of gradients in RNN can explode or vanish
    - Exploding gradients are controlled with gradient clipping
    - Vanishing gradients are controlled with additive interactions

> [!note]
> RNN's aren't as popular as they used to be and are now being replaced by attention, self-attention, and transformers. See [[Attention, Transformers]]
# Hierachial RNNs
These are RNNs that break a long sequence up into multiple chunks. Each chunk would then have it's own hidden state that summarizes its chunk. This way you can try to capture long-range information.