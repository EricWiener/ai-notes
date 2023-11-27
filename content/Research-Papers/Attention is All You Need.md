---
tags: [flashcards]
---

[[Attention Is All You Need, Ashish Vaswani et al., 2017.pdf]]

# Abstract
- The dominant sequence [[Transduction]] models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder.
- The best performing models also connect the encoder and decoder through an attention mechanism ([[Attention#Sequence-to-Sequence with RNN]]).
- This paper introduces the [[Transformer]].

# Introduction
- [[RNN, LSTM, Captioning|LSTM]] and [[GRU]] have been the state of the art in langauge processing.
- Sequence models are difficult to parallelize:
    - It's hard to parallelize between different examples because they ==can have different lengths==.
    - You can't parallelize within the same example because latter instances in the sequence ==rely on earlier instances==. 
> [!note]
> This paper proposes getting rid of recurrence and relying entirely on an attention mechanism to draw global dependencies between input and output.
<!--SR:!2024-09-11,673,310!2028-05-05,1755,350-->

# Background
- Current approaches to reducing sequential computation use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions.
    - The number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions (ex. linearly for [[ConvS2S-VC - Fully Convolutional Sequence-to-Sequence Voice Conversion]]) which makes it difficult to learn dependencies between distant positions.
    - I believe this is referring to stacking convolutional layers.

###  What is the purpose of Decoder mask (triangular mask) in Transformer?
[SO Post](https://ai.stackexchange.com/questions/23889/what-is-the-purpose-of-decoder-mask-triangular-mask-in-transformer)

the mask is needed to prevent the decoder from "peeking ahead" at ground truth during training, when using its Attention mechanism.

**Encoder:**
 - **Both runtime or training:** 
the encoder will always happen in a single iteration, because it will  process all embeddings separately, but in parallel. This helps us save time. 

**Decoder:**
 - **runtime:** 
Here the decoder will run in several non-parallel iterations, generating one "output" embedding at each iteration. Its output can then be used as input at the next iteration.
 - **training:** 
Here the decoder can do all of it in a single iteration, because it simply receives "ground truth" from us. Because we know these "truth" embeddings beforehand, they can be stored into a matrix as rows, so that they can be then submitted to decoder to be processed separately, but in parallel. 
As you can see during training, actual predictions by the decoder are not used to build up the target sequence (like LSTM would). Instead, what essentially is used here is a standard procedure called "teacher forcing".
As others said, the mask is needed to prevent the decoder from "peeking ahead" at ground truth during training, when using its Attention mechanism. 

As a reminder, in transformer, embeddings are **never** concatenated during input. Instead, each word flows through encoder and decoder separately, but simultaneously.

Also, notice that the mask contains **negative infinities**, **not zeros**. This is due to how the Softmax works in Attention.

We always first run the encoder, which always takes 1 iteration. The encoder then sits patiently on the side, as the decoder uses its values as needed.

# Transformer Architecture
![[Transformer#Encoder-decoder architecture]]

# Training
- Use dropout in multiple locations
- Use [[Label Smoothing|label smoothing]]. [[Label Smoothing]] can hurt [[Perplexity|perplexity]] as the model learns to be unsure, but can improve accuracy and [[BLEU scores]].

# Results
- Evaluated on English-to-German translation.
- Achieves better [[BLEU scores]] than previous state of the art models.
- Used [[Beam Search]] with length normalizing ala [[Google's Neural Machine Translation System_ Bridging the Gap between Human and Machine Translation, Yonghui Wu et al., 2016.pdf]]
- Also used [[Checkpoint Averaging]]
- Quality drops off with too many heads
- Reducing the attention key size $d_k$ hurts model quality.
- Bigger models are better.
- Dropout is helpful to avoid over-fitting.
- Learned positional embeddings have similar results to sinusoidal positional encodings.
- Generalized well to other tasks (tried out on English [[Contituency Parsing]])

# Additional Resources
- [Notebook showing an implementation with annotations](https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb)
- [Very good blog post from Machine Learning Mastery](https://machinelearningmastery.com/the-transformer-model/)
- [SO Post](https://stats.stackexchange.com/questions/508290/what-is-masking-in-the-attention-if-all-you-need-paper) that explains multiple parts of the diagram

# Questions
- In the background, what does the paper refer to with "recurrent attention mechanism instead of sequence-aligned recurrence"
- Paper mentioned exploring "restricted attention mechanisms to efficiently handle large inputs and outputs". What work has been done on this?