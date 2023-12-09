---
tags:
  - flashcards
source: 
summary: unfreeze more and more of the model's parameters over time.
publish: true
---
In gradual unfreezing, more and more of the model’s parameters are fine-tuned over time. Gradual unfreezing was originally applied to a language model architecture consisting of a single stack of layers. At the start of fine-tuning only the parameters of the final layer are updated, then after training for a certain number of updates the parameters of the second-to-last layer are also included, and so on until the entire network’s parameters are being fine-tuned. Source: [[Research-Papers/Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer|T5]]