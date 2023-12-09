---
tags:
  - flashcards
source: https://lightning.ai/pages/community/article/understanding-llama-adapters/
summary: 
aliases:
  - Prompt Tuning
  - Prefix Tuning
  - Soft Prompt Tuning
  - Hard Prompt Tuning
  - PEFT
---
PEFT is a way of fine-tuning pre-train models, but keeping most of the original weights frozen. This allows people to use pretrained models without requiring a lot of additional compute or resources.

Some approaches are:
- Prompt tuning (hard or soft)
- LoRA
- Adapter modules
- [[LLaMA-Adapter]]

# Adapting an LLM for a specific task
There are four possible approaches to adapt an LLM to a task:
- [[In-Context Learning]] where you provide examples to the model of what you want it to do and then it performs a task.
- Feature-based finetuning: you train a small auxiliary classifier on the output embeddings from the frozen transformer.
- Finetuning I: you add a couple output layers and train these (similar to training a logistic regression classifier or small MLP on the embedded features).
- Finetuning II: update all layers of the transformer. This is similar to Finetuning I but you don't freeze the parameters of the pretrained LLM.

The three finetuning approaches allow you to trade off between model performance and computational requirements/training budget:
![[understanding-parameter-efficient-finetuning-of-large-language-models-20231009114035984.png]]

# Parameter Efficient Finetuning (PEFT)
LLMs are very large and can be hard to fit into GPU memory on a single computer. Therefore, techniques were developed that only require training a small number of parameters but still get the benefits from Finetuning II.

One PEFT technique that is popular is LLaMA-Adapter which can be used to finetune [[LLaMA 2]]. [[LLama v2 Adapter]] makes use of two related techniques: [[Prefix Tuning]] and [[Adapters]].

# Prompt Tuning and Prefix Tuning
The original concept of prompt tuning refers to changing the prompt to the LLM to achieve better modeling results.

However, it now means adding learnable prompt tokens to pre-trained models, which are inserted either to input embeddings only or to multiple intermediate layers.
### Hard Prompt Tuning
The following is an example of hard prompt tuning to achieve a better translation result:
```
1) "Translate the English sentence '{english_sentence}' into German: {german_translation)"
2) "English: '{english_sentence}' | German: (german_translation}"
3) "From English to German: '{english_sentence}' -> {german_translation}"
```

Hard prompt tuning is where you directly change the ==discrete input tokens== which are not differentiable.
<!--SR:!2024-01-10,41,290-->

### Soft Prompt Tuning
Soft prompt tuning concatenates the embeddings of the input tokens with a trainable tensor that can be optimized via backprop to improve the modeling performance on a target task. You can train a small model for a specific task that learns to generate an output that performs the model's overall performance.
<!--SR:!2023-10-13,4,270-->

After learning a soft prompt, you have to supply it as a prefix when performing the specific task you finetuned the model on. This allows the model to tailor its responses to that particular task. Moreover, we can have multiple soft prompts, each corresponding to a different task, and provide the appropriate prefix during inference to achieve optimal results for a particular task.
### Prefix Tuning
Prefix tuning is a type of prompt tuning where you add a trainable tensor to the input to each transformer block (vs. just the input embeddings as in soft prompt tuning).

> [!NOTE]
> What's the difference between soft prompt tuning and prefix tuning?
??
Soft prompt tuning concatenates the embedding of the input tokens with a tensor learned by an auxiliary model for a specific task. Prefix tuning will concatenate the additional tensor to the input of each transformer block instead of just the embedding layer.
<!--SR:!2024-02-20,88,290-->

The following figure illustrates the difference between a regular transformer block and a transformer block modified with a prefix:
![[understanding-parameter-efficient-finetuning-of-large-language-models-20231009115629948.png]]

The fully connected layers embed the soft prompt in a feature space with the same dimensionality as the transformer-block input to ensure compatibility for concatenation.

The psuedo-code is below:
![[understanding-parameter-efficient-finetuning-of-large-language-models-20231009115755230.png]]

> [!NOTE]
> According to the original [prefix tuning](https://arxiv.org/abs/2101.00190) paper, prefix tuning achieves comparable modeling performance to finetuning all layers while only requiring the training of 0.1% of the parameters.

### Implementation Details**
The prefix $P_\theta$ is a matrix with dimensions (prefix_length x d) where $d$ is the hidden dimension size. For a prefix length of 10 and hidden size 1024, the prefix would contain 10,240 tunable parameters.

During training, the prefix values are optimized to maximize the likelihood of generating the correct output text y given input x. The gradients of the loss function are only computed with respect to the prefix parameters. The parameters of the pretrained model itself are kept completely fixed.

The authors found that directly optimizing $P_\theta$ is unstable, so they reparameterized it with a smaller matrix $P'_\theta$ which is a smaller matrix with size (prefix_length x k) where $k < d$. So $P'_\theta$ has fewer columns and is smaller than $P_\theta$.

This reparameterization with the smaller $P'_\theta$ acts as a “bottleneck” that helps stabilize optimization.

Then $P_\theta$ is computed from the smaller $P'_\theta$ by an MLP that expands it to the larger $d$: $P_\theta[i,:] = \text{MLP}(P'\theta[i,:])$. [Source](https://medium.com/@musicalchemist/prefix-tuning-lightweight-adaptation-of-large-language-models-for-customized-natural-language-a8a93165c132)

After training, only the final $P_\theta$ is needed.
# Adapter Modules
See [[Parameter-Efficient Transfer Learning for NLP|Adapter Modules]]

# LoRA
This approach introduces trainable, rank, decomposition matrices into each network weights. This has shown promising fine-tuning abilities on large generative models.