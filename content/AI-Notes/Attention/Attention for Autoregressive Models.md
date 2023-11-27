---
tags: [flashcards]
source:
summary: Attention for models that predict a sequence
---

[[Autoregressive]] models predict future values based on previous ones. In order to create autoregressive [[Transformer]]'s you need to use masked attention.

# Masked Attention Overview
See [[Masked Attention]] for a good explanation of why masked attention is needed.
- Masked attention is needed when working with sequences where the output at one state is the input to the model at the next state (this could be trying to generate text, translating, etc.). 
- The transformer outputs a probability distribuition after the softmax and it's possible you have multiple layers of transformers before you give the final output. Therefore, you have no way of knowing what the next output will be without going through each timestep sequentially. 
- The purpose of masking is that you prevent the decoder state from attending to positions that correspond to tokens "in the future", i.e., those that will not be known at the inference time, because they will not have been generated yet. [SO Post](https://datascience.stackexchange.com/a/81492/70970).
- Masking is used to avoid attending to succeeding words. [Source](https://machinelearningmastery.com/the-transformer-model/).
- During **training**, you can use [[Teacher Forcing]] and just give the ground truth as the inputs, so you only need to make one pass (the outputs aren't needed as inputs). However, since you are giving the entire ground truth as the input, you need to use masking to make sure the model doesn't  =="look ahead"== in the sequence and cheat (just generating whatever the next input is).
- During **inference**, you need to have the model generate output one step at a time (using the previous output as an input since you don't know what the correct prediction should be).
<!--SR:!2028-09-13,1857,350-->

**When to use masked self-attention vs.  masked attention**:
- **Masked attention**: if translating, you will have the entire document in your original language, but this likely won't be what you use as the query vectors (so self-attention doesn't make sense). For translation, you will likely have the input document as the keys/values and have the query vector at each time step be the previously predicted word in the other language (beginning with a `[START]` token).
- [[Attention#Self-Attention Layer]]: Self-attention is more applicable for something like text generation where you have the original document and you are trying to get your model to predict the same document.
- Ex: if you wanted to translate from English to French with an RNN and attention, you could use self-attention over the hidden states of the encoder (with no masking since you want to attend to all the words). Then, you could use a masked self-attention layer to process the decoder hidden state (since you want each word dependent on the words that came before it, but not after it).

# Masked Attention Implementation
You can put in a $- \infty$ in every position of the similarities (pre-normalization) in every vector position you want it to ignore. After it passes through soft-max, these will all have zero weight.

![[masked-self-attention.png]]
In the example above, when computing the output for the first input $Q_1$, we just want it to look at the first key, so we block off $K_2, K_3$. For $Q_2$, we let it look at both $K_1, K_2$. This is similar to how RNNs would only look at the previous states. We are now able to get rid of the RNN entirely and still handle sequences of data just with attention.

**Note**: the masking occurs at every layer if you have stacked attention layers - this ensures you don't attend to future information that got propogated from the previous layer. [Source](https://datascience.stackexchange.com/a/88128/70970)

**Note**: the above example shows masked attention used with self-attention, but the two can be used seperately. In the above example, we are doing something like document generation where you just want to predict the input text during training (so you can use [[Teacher Forcing]] to input the correct tokens regardless of what the model actually predicts).

**Additional Resources**:
- [This SO post about fairseq's implementation](https://datascience.stackexchange.com/a/65070/70970): "the mask is used to force predictions to only attend to the tokens at previous positions, so that the model can be used autoregressively at inference time. This corresponds to parameter `attn_mask`."