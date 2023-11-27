---
tags:
  - flashcards
source: https://arxiv.org/pdf/1803.02155.pdf
summary: 
aliases:
  - Relative Positional Encodings
---
[Good blog post](https://jaketae.github.io/study/relative-positional-encoding/)

### Reasons to use relative positional encodings over absolute
**Handling sequences of arbitrary length**
Using absolute positional information means that there is a **limit to the number of tokens a model can process**. If a language model can only encode up to 1024 positions, it can't process sequences longer than 1024 tokens.

Relative positional encodings can generalize to sequences of unseen lengths, since theoretically the only information it encodes is the relative pairwise distance between two tokens.

### Math
Relative positional encodings (as they were first introduced) add relative positional information when calculating attention by adding the information to the key and value matrices.

Later works only add the relative positional information to the keys.

