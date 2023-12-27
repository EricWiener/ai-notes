---
tags:
  - flashcards
source: https://lightning.ai/pages/community/article/understanding-llama-adapters/
summary: 
aliases:
  - PEFT
---
PEFT is a way of fine-tuning pre-train models, but keeping most of the original weights frozen. This allows people to use pretrained models without requiring a lot of additional compute or resources.

Some approaches are:
- [[AI-Notes/Large Language Models/Prompt Tuning and Prefix Tuning|Prompt tuning (hard or soft)]]
- [[Research-Papers/LoRA|LoRA]]
- [[Research-Papers/Parameter-Efficient Transfer Learning for NLP|Adapter Modules]]
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

One PEFT technique that is popular is LLaMA-Adapter which can be used to finetune [[LLaMA 2]]. [[LLama v2 Adapter]] makes use of two related techniques: [[AI-Notes/Large Language Models/Parameter Efficient Fine-Tuning|Prefix Tuning]] and [[Research-Papers/Parameter-Efficient Transfer Learning for NLP|Adapter Modules]].

### [[AI-Notes/Large Language Models/Prompt Tuning and Prefix Tuning|Prompt Tuning and Prefix Tuning]]

# Adapter Modules
See [[Parameter-Efficient Transfer Learning for NLP|Adapter Modules]]
# LoRA
This approach introduces trainable, rank, decomposition matrices into each network weights. This has shown promising fine-tuning abilities on large generative models.