---
tags:
  - flashcards
source: https://arxiv.org/abs/2210.03629
summary: Introduces the concept of
aliases:
  - ReAct
---
The goal of ReAct is to enable language models, such as ​GPT-like models, to not only understand and generate text but also to perform reasoning tasks and take actions based on the given context. The authors argue that by incorporating reasoning and acting abilities into language models, they can achieve more sophisticated and effective language understanding and generation.

ReAct introduces a two-step process for language models. First, it performs reasoning by using external knowledge sources, such as structured databases or knowledge graphs, to gather relevant information. This enables the model to reason over facts and make informed decisions based on the context.

The second step is acting, where the model generates language-based instructions or commands to interact with the environment or perform specific tasks. The authors propose a ​Reinforcement Learning (RL) framework to train the model to optimize its actions based on rewards or feedback.