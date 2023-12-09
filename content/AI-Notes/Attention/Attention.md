---
tags: [flashcards, eecs498-dl4cv]
summary: Self-attention is an alternative way than RNNs/LSTMs to process sequences of data
---

[This article looks good, but haven't read it yet](https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853)

> [!note] This page will walk through the development of the attention layer
> Attention first started as a concept for sequence-to-sequence models. However, it was later expanded to be used with non-[[Autoregressive|autoregressive]] models. For more details on using transformers with autoregressive models, see [[Attention for Autoregressive Models]] and [[Optimized Autoregressive Transformer]].

# Sequence-to-Sequence with RNN
> [!note]
> In [[RNN, LSTM, Captioning#Sequence to Sequence seq2seq]], we combined a many-to-one and one-to-many RNN in order to generate an image caption. We didn't use a context vector there and we also didn't pass the previous output as the input to the next layer for the decoder (we were using a one-to-many architecture for the decoder). Now that we want to introduce attention, we are going to be adding in a context vector.
In the RNN lecture we covered how you can convert between sequences using an RNN (ex. English to French translations). You have an **encoder** network encode the sequence into a single initial decoder state + context vector, and then you pass those onto the **decoder** network which can produce the output sequence.

- Inputs: sequence $x_1, ..., x_T$
- Outputs: sequence $y_1, ..., y_{T'}$ (can be different length)

### Encoder
Once you process the input sequence, you want to somehow summarize the entire content of the sequence into a **initial decoder state** $s_0$ and **context vector** $c$
![[seq-to-seq-rnn-encoder.png]]
- The decoder state $s_0$ is the first hidden state of the decoder. It is common to predict $s_0$ as the output of a fully connected network.
- The context vector $c$ is passed to every layer of the decoder. This could be set to the output of the final hidden state.

## Decoder
The decoder gives the next hidden state as a function of $s_t = g_U(y_{t-1}, h_{t-1}, c)$.
![[seq-to-seq-rnn-encoder-decoder.png]]
- The decoder receives $s_0$ as the initial hidden state
- It also passes the context vector $c$ to every layer.
- You then pass `<start>` as the starting token to kick-off the sequence generation until `<stop>` is an output.

**Context vector:** 
The context vector is responsible for passing information between the encoder and the decoder. It is supposed to ==summarize the entire input sequence==. This works alright if you are just doing a short sentence, but if you are doing a whole document, this doesn't work as well (**hard to encode an entire document in a ==single vector==**).
<!--SR:!2024-03-03,566,335!2029-08-04,2133,359-->

Additionally, in the case of a many-to-many problem where we have the same length input and output, we can often directly use the corresponding input hidden state for the context vector for the decoder. However, we run into issues if we have variable length inputs and outputs. We can no longer just pass a certain hidden state since there won't be a direct correspondence and all learned operations require matrix multiplication which expects fixed size vectors (you need to learn a fixed set of weights). Therefore, it is very challenging to have a ==learned method to compute a context vector== (i.e. more sophisticated than averaging the hidden states) given variable length inputs and/or outputs.
<!--SR:!2024-03-12,390,299-->

We can instead compute a new context vector at each step of the decoder using attention.

# Attention with RNNs

We are still using a sequence-to-sequence RNN, but we add something called Attention that allows the RNN to ==recompute== the context vector at each time-step of the decoder (each step of the decoder can compute a different context vector).
<!--SR:!2024-04-27,437,292-->

### Alignment Scores (aka Similarity Scores)
We will use an alignment function $e_{t, i} = f_{\text{att}}(s_{t-1}, h_i)$ that is a tiny fully-connected neural network. It will receive the current hidden state of the decoder and also one of the states of the encoder. It tells you an **alignment score** (un-normalized scalar) of how much you should ==pay attention to each state of the encoder== given the current state of the decoder.
![[alignment-scores.png]]
For instance, when looking at the first state of the decoder, $s_1$, we compute the following to see what hidden state of the encoder to pay most attention to:
<!--SR:!2024-03-11,574,339-->

- $e_{1, 1} = f_{\text{att}}(s_{0}, h_1)$
- $e_{1, 2} = f_{\text{att}}(s_{0}, h_2)$
- $e_{1, 3} = f_{\text{att}}(s_{0}, h_3)$
- $e_{1, 4} = f_{\text{att}}(s_{0}, h_4)$

<details><summary>Flashcard</summary>

What are alignment scores (aka similarity scores)?
?
Alignment scores are un-normalized scalars that tell you how much attention to pay to each hidden state of the encoder. They are the output of a tiny FC network that takes the current decoder state as input.
![[alignment-scores.png]]

</details>

### Attention Weights
However, currently we are outputting a scalar value. We want to normalize these into probabilities, so we can add a ==softmax on-top==. Now the attentions will sum to 1 and be between 0 and 1. These are called **attention weights** because they say how much we want to weigh each of the hidden states.
<!--SR:!2029-07-15,2114,359-->

![[attention-weights.png]]

You compute the context vector as a ==linear combination of hidden states==: $c_t = \sum_i a_{t, i}h_i$. The intuition behind this is that different parts of the input sequence have a different influence on the output sequence.
<!--SR:!2028-02-10,1670,339-->

![[context-vector-is-linear-combination.png|The context vector is a linear combination with weights $a$ multiplied by their corresponding hidden state $h$]]

The context vector is a linear combination with weights $a$ multiplied by their corresponding hidden state $h$ which allows the attention weights to change depending on the current state of the decoder. It is a learnable value that steers how much the decoder weights different encoder hidden states. 

Note that all the operations done are **differentiable**. We aren't telling it specifically what state to look at. You can backpropogate through everything.

> [!note]
> Now you can have a different context vector for each step of the decoder. The attention weights are calculated as the output of a small FC net that takes the current decoder state as input which allows the attention weights to change depending on the current state of the decoder. The input sequence is not forced to be crammed into a single vector.

You can visualize the attention weights that an RNN tasked with translating English to French generates. The horizontal axis shows the tokens of the encoder sequence. The vertical axis shows the tokens of the decoder sequence. The lighter the color in the grid, the higher the attention weights were for encoder hidden states for the corresponding output token (the rows will sum to 1). If there is a diagonal sequence, then the words correspond in order. If there not a diagonal, then attention is figuring out a different output word order from the input.

![[attention-weights-visualized.png]]
![[visualizing-english-to-french-translation-annota.png]]
# Generalizing Attention

The concept of attention isn't limited to just sequences. The decoder doesn't treat $h_i$ as an ordered sequence. It just sees it as an unordered set $\{h_i\}$ and then computes attention weights for these hidden states. You can use a similar concept given any sets of input hidden vectors $\{h_i\}$.

### Image Captioning with RNNs and Attention

![[image-captioning-w-attention.png]]

You can use attention when generating image captions.
- First you use a CNN to compute a grid of features for an image. We don't use a FC layer at the end, and instead use the final grid of outputs (vs. in attention for with RNNs where we passed the hidden states through a FC). We will consider this a grid of feature vectors.
- Then you use the grid of vectors to predict the initial hidden state $s_0$. This can be done by flattening the grid and running it through a FC network.
- You can then calculate attention scores for the grid of feature vectors using $s_0$
- You then have to use `softmax` to normalize these alignment scores
- You then get your attention weights such that the entire grid of attention weights sums to 1.
- You can use your attention weights to compute your context vector $c_1$ where $c_t = \sum_{i,j} a_{t, i, j} h_{i, j}$ (the context vector is a weighted sum of the attention weights and the elements in the feature grid).
- You combine $c_1$ with $y_0$ (in this example it is the embedding for `[START]`) to get your next hidden state $s_1$.

You can then repeat this process over and over for every word you predict.

![[visualizing-image-captioning-attention.png]]
Above is a visualization of what image areas are weighted more for each word.

#### Similarity of Attention to the human eye:
The human retina has the highest acuity (best perception) in one specific region called the **fovea**. In the other regions, the perception isn't as good. To make up for this, the human eye moves constantly (called **saccades**). The attention weights at each timestep are kind of like saccades of human eye.

# Attention Layer
> [!note]
> 
> Attention started off as a way to avoid the bottleneck issue with RNNs (trying to contain all information from the input sequence into a single vector), but people realized it could be used for a variety of tasks.

## Generalizing notation
We want to generalize the attention layer so we can generalize it to a new multi-purpose layer. We can generalize the notation as follows:
![[generalizing-attention-layer-step-one.png]]
**Inputs:**
- **Query vector:** $q$ (shape $D_Q$) replaces the decoder hidden states that we had from each step of the decoder ($s_i$). For instance, this would replace $s_0$ or $s_1$.
    - $D_Q$ refers to how many dimensions will be in our query vector (not how many query vectors we have).
    - We have one query vector for an input. However, if we embed our input vector, it might have a different shape after the embedding than the original input. For instance, converting a one-hot vector for a vocabulary of 10,000 words into a much denser vector of length 1,000. 
    - **Note**: when we generalize computation later on and use the dot-product for similarity score, the query vector will have to have the same dimensions as the input vectors ($D_Q = D_X$).
- **Input vectors:** $X$ (shape $N_x \times D_x$) is the set of feature vectors that we want to attend over. There are $N_x$ vectors each with $D_x$ dimensions.  Previously this was the hidden state (shown in blue in the diagram as $h_{1, 1} \dots h_{3, 3}$).
- **Similarity function** $f_{\text{att}}$

**Computation:**
- **Similarities:** e (shape $N_x$) $e_i = f_\text{att}(q, X_i)$. These are all scalars. These are the **same as the attention scores we used earlier.**
- **Attention weights:** $a = \text{softmax}(e)$ with shape $N_x$. These are the normalized weights.
- **Output vector:** $y = \sum_i a_iX_i$ with shape $D_x$. This is the weighted combination of the $N_x$ vectors in X. These replace the context vectors that we had seen (shown in purple in the above diagram as $c_1, \dots c_t$).

> [!note]
> $N_x$ refers to the number of input vectors in the set of input vectors. However, this is still a single training example - not a batch of data.
## **Generalizing computation:**
We will need to make some changes to our computation to generalize it:

**1. Use scaled dot product for similarity:**
![[attention,-transformers-20220222070100193.png]]
Instead of using a neural network (as early versions of attention did) for $f_{\text{att}}$ in $e_{i}=f_{a t t}\left(q, X_{i}\right)$, we will just use a dot product which works just as well in practice. Now, $e_i = q \cdot X_i$. However, we will also add a scaling term, $e_i = q \cdot X_i / \sqrt{D_q}$. This is called the ==scaled dot product== and is used to normalize the dot product for high dimensional vectors.
<!--SR:!2024-03-10,573,339-->

Note: previously we computed $e_{i}=f_{a t t}\left(q, X_{i}\right)$ where $f_{att}$ was a neural network that received a combined vector of the current hidden state of the decoder and the particular feature from the encoder.

> [!info]- Scaled dot product normalizes dot product for high dimensional vector (reasoning used in the paper)
> 
> This is because we are going to pass the similarity scores through a soft-max function. If one similarity score is very high, then it will cause a peaked distribution and make the other scores very small. Large similarities cause softmax to saturate and give vanishing gradients. Also, as we consider vectors of high dimensions, their dot products will likely have larger magnitudes.
> 
> - When computing dot product, it is equal to the magnitude of the two vectors multiplied with the angle between the vectors: $a \cdot b = |a||b|\cos(\theta)$.
> - If both $a$ and $b$ are constant vectors of dimension $D$
> - Then $|a| = (\sum_ia^2)^{1/2} = a \sqrt{D}$, which means the magnitude of $a$ is equal to $a$ multiplied by the square root of its dimension.
>     - This is because if we assume $a = [c, c, c, c, .... c, c] \in R^{D}$, then $|a| = (\sum_ia^2)^{1/2} = (c^2 + c^2 ... + c^2)^{1/2}$
>     - Which is the same as $(D*c^2)^{1/2} = c\sqrt{D}$
> - We divide the dot product by the square root of the dimension to counter-act this scaling.

> [!info]- Scaled dot product maintains variance of 1 (additional reasoning to explain why dot product gets large)
> 
> **Scaled dot-product attention** is an attention mechanism where the dot products are scaled down by $\sqrt{d_k}$. Formally we have a query $Q$, a key $K$ and a value $V$ and calculate the attention as:
> 
> ${\text{Attention}}(Q, K, V) = \text{softmax}(\frac{QK^{T}}{\sqrt{d_k}})V$
> 
> If we assume that $q$ and $k$ are $d_k$-dimensional vectors whose components are independent random variables with mean $0$ and variance $1$, then their dot product, $q \cdot k = \sum_{i=1}^{d_k} u_iv_i$, has mean $0$ and variance $d_k$. Since we would prefer these values to have variance $1$, we divide by $\sqrt{d_k}$.
> 
> In general, this normalization helps create more stable gradients. The dot product in the numerator could potentially reach large magnitudes, which could create extremely small gradients after a softmax operation. Scaling down this dot product is done to counteract this issue.

Now that we are using the dot product, we also need to require that the input vector and query vector have the same dimension, $D_Q$. We still have one query vector (which corresponds to the hidden state of the decoder - shown in yellow), but now it has to have the same dimensions as the set of input vectors (shown in blue in the diagram).

**Note**: some papers also use additive attention. This is "similar in theoretical complexity", but dot-product attention is much faster and space-efficient in practice since it can be implemented with optimized matrix multiplication.

**2. Use multiple query vectors**

![[attention-layer-using-multiple-query-vectors.png]]

Previously we only had one query vector (hidden state of the decoder) at each time. Now, we want to allow ==multiple query== vectors. When using the RNN, we could only process one query vector at a time since we could only compute one state of the RNN at a time (since each step relies on the previous step). However, now that we are generalizing the attention layer, we can allow for passing in multiple query vectors at once since we won't always be using attention with a ==sequence of data== (we can still pass a single query vector at a time if we wanted - but we can also support passing multiple). $Q$ will now be $N_Q \times D_Q$.
<!--SR:!2028-09-23,1867,352!2024-03-09,572,339-->

> [!note]
> $Q$ now has dimension $N_Q \times D_Q$. Note that the subscript $Q$ refers to these dimensions being for the query vector. However, note that you can have $N_Q \neq D_Q \neq N_x$. Although the subscripts are the same, the variables don't have to have the same value.

When computing similarities, we will use a matrix multiply to efficiently compute the dot products $E = QX^T / \sqrt{D_Q}$ (shape $N_Q \times N_x$), which is the same as $E_{i, j} = Q_i \cdot X_j / \sqrt{D_Q}$.

Attention weights will be computed using the `softmax(E, dim=1)`, which is shape ($N_Q \times N_X$).

The output vectors will be $Y = AX$ with shape $N_Q \times D_X$ and $Y_i = \sum_j A_{i, j}X_j$.

**3. Separating key and value vectors.** 
![[attention-layer-seperate-key-value.png]]

When you do a Google search for "How tall is the Empire State building," you don't want the results to actually contain the query "How tall is the Empire State building." Instead, you want the answer to your query. 

Right now we are using the input vectors $X$ in two different ways. You first compare the input vectors with each query vector to compute attention weights. Then, you use it again to produce the final output.

We can separate the input vectors $X$ into ==key vectors and value vectors==. This way we look what best matches our key vectors queries and calculate attention scores. Then, we can weight our value vectors by these attention scores. This helps us because what we want to ==look-up doesn't necessarily match the value of the data==, as with the Google query.
<!--SR:!2029-06-22,2091,359!2024-09-26,625,299-->

![[seperating-key-value-vectors.png]]
These matrices will transform the input into two new sets of vectors: one of keys and one of values. When computing the similarities we now use the key vectors. When computing the output vectors, we use the value vectors:
- You have a learnable key matrix $W_K$ with shape $D_X \times D_Q$. 
    - You compute the key vectors with $K = XW_k$ which results in keys of shape $N_X \times D_Q$. 
    - You then compute the similarity scores with the query vectors and the key vectors.
- You also have a learnable value matrix $W_V$ with shape $D_X \times D_V$. 
    - You compute the value vectors with $V = XW_V$ which results in shape $N_X \times D_Q$. 
    - You then compute the output vectors using the value vectors weighted by the attention weights.

## Attention Layer Diagram
- You receive input vectors $X_i$. For each input vector, you compute a key vector $K_i$ (using the weights $W_k$) and value vector $V_i$ (using the weights $W_v$).
- You receive a set of query vectors $Q_i$
- You then compare each ==key vector with each query vector==. This gives you a matrix of ==un-normalized similarity scores==. Each element ($E_{i, j}$) is the scaled dot product of one of the key vectors and one of the value vectors.
    
    ![[AI-Notes/Attention/attention-srcs/Screen_Shot 9.png]]
    
- You then ==normalize== the similarity scores. You perform softmax along the vertical direction (each column) to get the alignment scores. Each column of the normalized matrix gives you a probability distribution over all the inputs $X_1, X_2, X_3$ (each column adds up to 1, so for each query, you have weights for each key).
    
    ![[AI-Notes/Attention/attention-srcs/Screen_Shot 10.png]]
    
- You then use the value vectors to compute a ==weighted combination of the value vectors according to the alignment scores==. You would do $V_1 * A_{1, 1} + V_2 * A_{1, 2} + V_3 * A_{1, 3}$ and then repeat for each of the columns. This produces one output vector for each query vector $Q_i$ where the outputs are determined by the value vectors weighted by the normalized results of the key vectors dot producted with the query vectors. In other words: each output vector corresponds to a query vector and is a weighted combination of value vectors (where the weights tell you how much to weight the value vector based on the similarity between the key and query vectors).
    
    ![[attention-layer.png]]
    
    You can now use the Attention Layer whenever you have one set of queries and one set of outputs, then you can use this attention layer.
    
<!--SR:!2025-06-30,932,332!2024-11-06,479,315!2024-08-11,314,365!2024-08-10,313,365-->

## Self-Attention Layer
> [!NOTE] Self-Attention summary from [[Research-Papers/Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer|T5]]
> Self-attention is a variant of attention that processes a sequence by replacing each element by a weighted average of the rest of the sequence. 

This is a special case of the attention layer where you ==only have inputs and no queries==. 
<!--SR:!2027-09-01,1544,330-->

![[attention,-transformers-20220223152155333.png]]

You now need to make the input vectors the query vectors (which can be done using a new learnable weight matrix $W_Q$), as well as compute the key and value vectors from them. The original attention layer is shown on the left and the new self-attention layer is shown on the right. Note that the query vectors are now computed using a learned transformation applied to the input vectors.

> [!info]- Why do we have a seperate $W_K$ and $W_Q$ (vs. just $W_E$)?
> In the above computation, we have $Q = XW_Q$, $K = XW_K$, and then do $E = QK^T$ and don't use $Q$ or $K$ anywhere else. Therefore, we could theoretically just have a single learnable matrix that directly transforms $X$ into $E$ and not have a seperate $Q$ and $K$. However, the benefit of having the seperate $Q$ and $K$ is to reduce the dimensionality. If you just had a matrix $W_E$, then it would have to be size $D_Q \times D_Q$. $D_Q$ could be very large (many 1,000s) and $W_E$ would be very large. $D_K$ could be a set to be smaller than $D_Q$ to save the number of weights that need to be learned.

![[self-attention-layer.png]]

This layer receives a set of vectors and produces a set of vectors. Internally, it is comparing all the vectors in a non-linear way the network decides itself.

Note that if you change the order of the input vectors, the queries and keys will be the same, but in a different order. The similarities will also be the same (it's just a scaled dot product), but in a different order. The output vectors will all be the same, but in a different order. The weight matrices are applied 

The self-attention layer is ==permutation invariant==. This means f(s(x)) = s(f(x)) if $s()$ permutes the inputs and $f()$ is the self-attention layer. If you take the input vectors and apply a permutation on them, the output would be the same as if we didn't permute the inputs and then permuted the outputs.
<!--SR:!2024-11-14,737,310-->

> [!note]
> This is a new type of neural network layer that doesn't care about the order of the vectors.
In some cases, you might want the model to know the order of a vector. For instance, in translation, the subject is more likely to be at the beginning of the sentence.

You can recover some sensitivity to order by appending the input vector with some way to distinguish parts of the sequence (called **positional encodings**). This breaks the permutation equivariance of the model since the inputs will change dependening on their order (since the positional encodings will change).

The number of inputs is equal to the number of outputs.

> [!note] Attention models can handle sequences of any length
> 
> The length of the input is given by $N_X$ in the above diagram and this dimension doesn't affect anything else in the model. The reason that sequences are often forced to have the same length is in order to support batching multiple examples together.
**Note on using bucketing to avoid wasted memory**:
In theory, the attention mechanism can work with arbitrarily long sequences. The reason is that batches must be padded to the same length.

> Sentences of the maximum length will use all the attention weights, while shorter sentences will only use the first few.

By this sentence they mean they want to avoid batches like this:

```
A B C D E F G H I K L M N O
P Q _ _ _ _ _ _ _ _ _ _ _ _
R S T U _ _ _ _ _ _ _ _ _ _
V W _ _ _ _ _ _ _ _ _ _ _ _ 
```

Because of one long sequence, most of the memory is wasted for padding and not used from weights update.

A common strategy to avoid this problem (not included in the tutorial) is _bucketing_, i.e., having batches with an approximately constant number of words, but a different number of sequences in each batch, so the memory is used efficiently.

[Source](https://stats.stackexchange.com/a/411919/271266)

![[Multihead Self-Attention]]

### Example: CNN with Self-Attention

![[cnn-w-self-attention.png]]

First you run an image through a CNN. You get a set of features $C \times H \times W$. You then run these through 1x1 CONV to get separate matrices for queries, keys, and values.

![[AI-Notes/Attention/attention-srcs/Screen_Shot 16.png]]

You then compute the attention weights using the queries and keys. This tells us for every position in the input image, how much we want to attend to the other regions in the image.

![[AI-Notes/Attention/attention-srcs/Screen_Shot 17.png]]

You then use the attention weights and multiply with the values to get a weighted linear combination of the value vectors. You get a value vector for each position in the input.

Now, you get a new grid of feature vectors. Every position in the output grid depends on every feature in the input grid.

![[self-attention-module.png]]



Sometimes you also add an additional $1 \times 1$ convolution and a residual connection. This forms the entire contained self-attention module.

# Processing Sequences

![[sequence-processing-architectures-comparison.png]]

### [[RNN, LSTM, Captioning|RNN]]: works on Ordered Sequences
- Good at handling long sequences. The final output state can summarize the entire sequence.
- However, you can't ==parallelize them==. The hidden states are computed sequentially. You can't use GPUs to accelerate this very well.
<!--SR:!2029-05-30,2068,355-->

### 1D [[Convolutional Neural Networks|CONV layer]]: works on Multidimensional Grids
- Note these refer to a single 1 dimensional sliding [[Convolutional Neural Networks|CONV layer]] (not a [[Pointwise Convolution|1x1 Conv]]).
- Bad at long sequences. You need to ==stack== many CONV layers in order to get a receptive field that sees the whole sequence.
- Highly parallel: each output can be computed in parallel.
<!--SR:!2024-02-27,561,332-->

### Self-Attention: works on Sets of Vectors
- Good at ==long sequences==: after one self-attention layer, each output will depend on each input. The **path length is reduced significantly between long-range dependencies**.
- Highly parallel. Each output can be computed in parallel (during training).
- Computationally faster than recurrent layers (when sequence length < representation dimensionality). For handling sequences, it is faster than regular [[Convolutional Neural Networks|CONV layer]] and self-attention + point-wise feed-forward layer has same complexity as [[Depthwise Separable Kernels]] (this is only applicable to sequences - a single CONV layer is faster than a single attention layer).
- Yields more interable results: can visualize results.
- Con: Very memory intensive (need a key, value, and query weight matrices + input embedding, positional encoding, and 1x1 conv + the intermediate values).
<!--SR:!2027-03-06,1406,339-->

If you want to process sequences with NN, you usually do this with self-attention. Need to build a new primitive block type called the [[Transformer]].

# [[Attention for Vision]]
