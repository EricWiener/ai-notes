---
tags: [flashcards]
aliases: [Contrastive Captioners are Image-Text Foundation Models]
source: https://arxiv.org/abs/2205.01917
summary:
---

GIT is a Transformer decoder conditioned on both CLIP image tokens and text tokens. The model is trained using "teacher forcing" on a lot of (image, text) pairs.

The goal for the model is simply to predict the next text token, giving the image tokens and previous text tokens.