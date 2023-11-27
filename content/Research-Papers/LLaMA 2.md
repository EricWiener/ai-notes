---
tags:
  - flashcards
source: 
summary: 
aliases:
  - Open Foundation and Fine-Tuned Chat Models
---
[YouTube Overview](https://youtu.be/zJBpRn2zTco) (mostly fluff stuff and not implementation).

Trained with 2 trillion tokens (vs. 1.3 trillion for v1) and double the context length (4096 vs. 2048).

You can use LLaMA 2 for commercial use if your company had less than 700 million active monthly users before July 18, 2023.

# Implementation
The paper introduces ghost attention where the model pays attention over multiple turns of the conversation.

> [!QUESTION] Look into this more.

# Pre-training
Used more robust data cleaning and trained on 40% more tokens. They upsampled the most factual sources and didn't use any of Meta's data. They used a new mix of publicly available data.

# Reward Modeling
They used lots of reward modeling with human feedback.

They have two different reward models: one for helpfulness and one for safety.

The reward models (the models that evaluate how good a response is) were initialized from pretrained chat model checkpoints to ensure both models benefit from knowledge acquired during pretraining. The reward model "knows" what the chat model knows which helps prevents situations where one model has more information which could result in favoring hallucinations.

# Results
Out performs other open source models and performs better than Llama-1. Not amazing at coding though.

They said they performed on-par with ChatGPT3, but they didn't evaluate on code or reasoning-related prompts.

Llama 1 out performed v2 on the [[Social IQA]] dataset which tests commonsense reasoning.

Llama 1 also outperformed v2 on [[BoolQ]].

They saw models pick up a general notion of time. They also saw that Llama v2 picked up a right wing sentiment.