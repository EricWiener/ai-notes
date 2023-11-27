---
tags:
  - flashcards
source: https://youtu.be/bZQun8Y4L2A
summary: Talk by Andrew Karpathy on GPT and Transformers
---
# Summary
- The video discusses the training and application of large language models, specifically ​GPT (​Generative Pre-trained Transformer) models.
- GPT models go through stages of pre-training, supervised fine-tuning, reward modeling, and reinforcement learning.
- ​Pre-training involves gathering a large dataset and tokenizing it for the neural network to train on.
- ​Supervised fine-tuning involves collecting high-quality prompt-response data and training the model on it, allowing efficient fine-tuning for various downstream tasks.
- ​Reward modeling involves training a reward model to score the quality of completions for a prompt.
- Reinforcement learning uses the reward model to guide the model's generation during training, resulting in better performance.
- Fine-tuning and RLHF (Reinforcement Learning from Human Feedback) may lead to improved model performance.
- Effective prompts, tool use, and retrieval-augmented models can enhance the model's capabilities.
- Prompt engineering, constraint prompting, and tree search algorithms can optimize model outputs.
- Limitations and potential biases in LLMs are discussed, emphasizing the importance of human oversight and low-stakes applications.

# Highlights
- GPT-3 was trained with 175B parameters and trained on 300B tokens. LLaMA had 65B parameters and trained on 1-1.4T tokens and it is a stronger model. The amount of data trained on matters a lot more than number of parameters.
- Training LLaMA base model took 21 days, 2,048 A100 GPUs, and 5 million dollars.
# GPT Assistant Training Pipeline
![[State of GPT _ BRK216HFS 0-58 screenshot.png]]
Pre-training is where 99% of the training compute and FLOPs. 1000s of GPUs and months of training. The other three stages are fine tuning and use a lot less compute.
### Pre-training
You first need to gather a large amount of data (ex. Wikipedia, Stack Exchange, etc.). You then compute a lossless tokenization to convert words into integers which the transformer operates on. One typical algorithm is [[Byte Pair Encoding]].

![[state-of-gpt-20231002092453867.png]]

**Creating a batch**
Each document is a different length, so they add an end of text token to signify when document ends and the next begins. They then arrange the documents in a sequence separated by these tokens and wrap the sequence to a particular context length (`T=10` in example below) to form rows of the batch (batch size `B=4` below).

![[state-of-gpt-20231003090243937.png]]

**Training the transformer**
![[state-of-gpt-20231003090402083.png]]
Each cell only has context from the cells in its row and before it (to the left of it). The model is tasked to predict the next cell.

In the example above:
- green: a random highlighted token.
- yellow: its context
- red: its target

The model will output a probability distribution over the entire vocabulary (ex. 50k bins).

### Base models
The models learn a very powerful general representations that you can fine tune for multiple downstream tasks.

Previously to do sentiment analysis you would need to collect a large dataset of positive/negative examples. Now you can just train a large language model and just fine tune with a couple examples.

However, even better than fine tuning, you can use prompts to ask your model questions. During training you can insert questions and answers about a document and then the model will learn to answer questions. This realization happened with GPT-2.

Base models are not assistants and they don't answer questions. They just complete documents (ex. if you give it a question it will just give you more questions). However, you can trick it into following a prompt:
![[state-of-gpt-20231003091830593.png]]

# Supervised Finetuning (SFT)
You collect small, but high quality documents. You ask contractors to write lots of documents of the form (prompt, response).

You are changing out the dataset from a low quality, high quantity internet dataset with a high quality, low quantity document dataset.

# Reward Modeling
You shift your data collection to be of the form of comparisons. You have the same prompt and different outputs from the model. You then ask users (or contractors) to rank which response was better:
![[State of GPT _ BRK216HFS 13-54 screenshot.png]]

You then add a `reward` token at the end of each response and you train the model to predict the reward for all the different prompts. You then supervise these predictions with the GT rankings and this way the model learns to predict how good a completion actually is.
![[State of GPT _ BRK216HFS 14-5 screenshot.png]]
In the above example, the blue tokens are the same prompt across each row. The yellow tokens are the completions from the SFT model. You then add a green reward token to the end of the completion.

# Reinforcement Learning
You now do reinforcement learning with respect to the reward model (you freeze the reward model).

![[State of GPT _ BRK216HFS 15-50 screenshot.png]]
You now train on the yellow tokens (the completions from the model initialized from the SFT model) and you use the frozen RM model to tell you how good each response was.

You now weight the language modeling objective by the rewards. If the completion was good, each token in the row gets its probability boosted. If the completion was bad, each token in the row gets its probability decreased. You do this on many batches and many rows and you end up with high scoring completions.

You then end up with an RLHF (reinforcement learning with human feedback) model. This is what ChatGPT is. Humans prefer the completions from RLHF compared to SFT and base models.

One reason why RLHF models could be better is because it is easier to discriminate (say which response is better) than to produce a response. This makes it more straightforward to use human evaluation and therefore you get a better GT. However, you might end up with [[Generative Adversarial Networks#Mode collapse|mode collapse]] (less diverse outputs). Base models might be better with tasks where you have N examples of things and want to generate more things (ex. generating Pokemon names).

# How to apply GPT assistants
LLMs don't have the internal thought process that humans have when writing. Instead, they just spend the same amount of compute on every token. The agents don't know what they are good at or not good at. They don't sanity check or correct as they go - they just sample token sequences.

They do have a huge number of parameters which allows it to access a huge amount of stored facts. It also has a perfect (finite) working memory. Whatever fits into the transformer context window is immediately available to the transformer.

Prompting is making up for the difference between human and LLM brains. LLMs are more like system 1 thinking (near-instantaneous process; it happens automatically, intuitively, and with little effort). System 2 thinking is slower and requires more effort. It is conscious and logical. [Source](https://thedecisionlab.com/reference-guide/philosophy/system-1-and-system-2-thinking#:~:text=System%201%20thinking%20is%20a,It%20is%20conscious%20and%20logical.).

### Chain of thought
> [!NOTE] Models need tokens to think

You can break up a complex task into multiple parts and prompt the models to have internal monologues to spread out the reasoning over more tokens. You can get the transformer to do this by prompting with things like "show this step-by-step". This results in the transformer needing to do less computational work per token so it can "think" things through.

### Ensemble multiple attempts (sanity check)
You can have the model generate N answers to the same prompt. You can then select the best answer from a series of attempts and then continue with the best answer. This allows the model to recover from its mistakes.

### Condition on good performance
Transformers want to imitate training sets that have a variety of performance qualities. You should prompt the model to be an expert ("you are a leading expert in this topic", "pretend you have an IQ 120", etc.). This will make the model output better results.

### Tool use / Plugins
You can offload tasks that LLMs are not good at. You can mix text with special tokens that call external APIs (ex. calculators, code interpreters, etc.).

The LLM doesn't know what it doesn't know and doesn't know about the tools. You need to tell it what it isn't good at and how to use the tools (ex: "you are not good at mental arithmetic. Whenever you need to perform calculations, use this token sequence to use a calculator").

### Retrieval Augmented LLMs
You can load related context/information into working memory context window.

Recipe:
- Break up the relevant document into chunks
- Use embedding APIs to index chunks into a vector store (ex. LlamaIndex).
- Given a test-time query, retrieve related information
- Organize the information into the prompt

### Constrained Prompting
Techniques for forcing a template in the output of LLMs. Ex: you can specify "```json```" and the probabilities of the model will be clamped to outputting valid JSON.

### Finetuning
Finetuning can also be used in addition to prompt engineering. This will actually change the weights of the model.
- Parameter Efficient FineTuning (PEFT), like LoRA, let you train small, sparse pieces of the model. Most of the model is frozen, but you change a small part and this works well in practice.
- You can use low-precision inference (ex. bitsandbytes) for the frozen parts of the model during finetuning since the frozen parts don't matter for gradient descent so you don't care about the values so much.
- You can start with open-source high quality base models (ex. LLama).

> [!NOTE] Finetuning is a lot more complicated than prompting. RLHF is research territory and is very hard to get to work. SFT is more doable on your own.

# Default Recommendations
Break up your task into two major goals: achieve your top possible performance and optimize your performance.

**Achieve your top possible performance:**
- Use GPT-4
- Use prompts with detailed task context, relevant information, instructions. Think along the lines of if you were assigning the task to a contractor and they can't email you back for more info.
- Retrieve and add any relevant context or information to the prompt
- Experiment with prompt engineering techniques (previous slides)
- Provide the LLM with a few examples that are 1) relevant to the test case, 2) diverse (if appropriate)
- Experiment with tools/plugins to offload tasks difficult for LLMs (calculator, code execution, etc.)
- Spend quality time optimizing a pipeline / "chain"
- If you feel confident that you maxed out prompting, consider SFT data collection + finetuning
- Expert / fragile / research zone: consider RM data collection, RLHF finetuning

**Optimize costs:**
- Once you have the top possible performance, attempt cost saving measures (e.g. use GPT-3.5, find shorter prompts, etc.)

### Recommendations
- Use in low-stakes applications, combine with human oversight
- Source of inspiration, suggestions
- Think of them as Copilots and not autonomous agents