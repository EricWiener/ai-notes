---
tags:
  - flashcards
source: https://huggingface.co/blog/moe
summary: 
aliases:
  - MoE
---
Mixture of Experts are bigger models comprised of multiple smaller blocks or experts. When tokens come in they are dynamically rooted to a subset of the experts so the entire model isn't active during inference. This makes the model lightweight and effective.

### Motivation
From [[Research-Papers/Scaling Laws for Neural Language Models|Scaling Laws for Neural Language Models]] we know that training a larger model for fewer steps is better than training a smaller model for more steps. MoE enables models to be pretrained with less compute which means you can scale up the model or dataset size with the same compute budget as a dense model (a traditional non-MoE model). A MoE model should achieve the same quality as its dense counterpart much faster during pretraining. [Source](https://huggingface.co/blog/moe).

### What is a MoE?
In the context of transformer models, a MoE consists of two main elements:
- Sparse MoE layers are used instead of dense feed-forward network (FFN) layers. MoE layers have a certain number of “experts” (e.g. 8), where each expert is a neural network.
- A gate network or router, that determines which tokens are sent to which expert. The router is learned at the same time as the rest of the network.

![[AI-Notes/Concepts/assets/Mixture of Experts/image-20240112102206930.png]]
MoE layer from the [Switch Transformers paper](https://arxiv.org/abs/2101.03961)

### Challenges
- Training: MoEs enable significantly more compute-efficient pretraining, but they’ve historically struggled to generalize during fine-tuning, leading to overfitting.
- Inference: Although a MoE might have many parameters, only some of them are used during inference. This leads to much faster inference compared to a dense model with the same number of parameters. However, all parameters need to be loaded in RAM, so memory requirements are high.

