---
tags:
  - flashcards
source: https://arxiv.org/abs/2302.13971
summary: You can train smaller LLMs for longer and using high quality data to get a good model
aliases:
  - Open and Efficient Foundation Language Models
---
[YouTube Overview](https://youtu.be/E5OnoYF2oAk)

Main idea: train models that are cheaper at inference (a smaller model trained longer will be cheaper at inference although training may be more expensive).

The [[Chinchilla]] papers focused on if you have a fixed training budget, how do you allocate it the best.

LLaMA-13B outperforms GPT-3 (175B parameters).
### Dataset
The paper has a large emphasis on how they open-source their work. They use only open-source datasets.

> [!NOTE] 
> The YouTube author had an interesting observation that using datasets like Wikipedia and Books might just boost results on evaluation tasks since you memorize tasks, but might not necessarily result in a stronger model when it comes to non-fact based responses.

The paper uses [[Byte Pair Encoding]] to encode the dataset. However, they split all numbers into individual digits to avoid numbers like `858` being formed by the learned tokens `5, 8, 58`. This is because it makes arithmetic very difficult to do since you need to learn to do arithmetic with all possible pairs of byte pair encodings. This might be a downside for dealing with things like years (ex. 1999) since humans recognize years as a whole concept.

The entire training dataset contains roughly 1.4T tokens after tokenization. For most of our training data, each token is used only once during training, with the exception of the Wikipedia and Books domains, over which they perform approximately two epochs.
### Hyperparameters
 ![[llama-hyperparameters.png]]
 It's interesting to see they use a batch size of 4 million.
### Training Details
- They use [[Learning Deep Transformer Models for Machine Translation|Pre-Norm]] which moves the [[Layernorm]] activation before the residual connection so the gradient can flow freely without needing to go through the Layernorm block again.
- They use the [[GLU Variants Improve Transformer|SwiGLU]] activation function to replace ReLU.
- They use [[Rotary Positional Embedding]] instead of absolute positional embeddings.
- They use [[AdamW]] as the optimizer with gradient clipping of 1.0 (you set gradients element-wise to `grad = max(grad, 1)`).

### Efficient Implementation
**Improved Causal MHSA**
The use an efficient implementation of causal (you are predicting the next token in a sequence based only on previous tokens) multi-head attention to reduce memory usage and runtime. They use the `xformers` library which doesn't store the attention weights + not computing key/query scores that are masked during the language modeling task.

**Activation Checkpointing**
They save the activations that are expensive to compute (ex. output of linear layers) by manually implementing the backward pass vs. relying on PyTorch autograd. This trades off faster training for lower memory usage.

### Results
**Multi-Shot QA**
They outperform or are on-par with models that are bigger than they are when evaluated on question answering [[Few-Shot Learning]] challenges.

**Massive Multitask Language Understanding**
They perform worse than the PaLM model for Massive Multitask Language Understanding which is a benchmark consisting of multiple choice questions covering various domains of knowledge, including humanities, STEM, and social sciences. They gave the potential explanation that they have used a limited amount of books and academic papers in their pre-training data, i.e., ArXiv, Gutenberg and Books3, that sums up to only 177GB, while the other models were trained on up to 2TB of books.

**Toxicity Measurement**
They measure how toxic the model is using the [PerplexityAPI](https://perspectiveapi.com/). They have a basic model and a "respectful" model whose prompts begin with "Complete the following sentence in a polite, respectful, and unbiased manner."

For the largest model it becomes more toxic as it becomes more respectful.

**[[WinoGender]] Dataset**
The model was evaluated on this task to see how biased the model was for gender.

**Carbon Footprint**
They calculate the carbon footprint of their model. Their largest model emitted 173 tons of CO2 (about the same produced as 173 individual passengers generate flying to Europe).