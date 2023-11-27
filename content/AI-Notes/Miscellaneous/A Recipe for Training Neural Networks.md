---
tags: [flashcards]
source: http://karpathy.github.io/2019/04/25/recipe/
summary:
---

> [!NOTE]
> **A “fast and furious” approach to training neural networks does not work** and only leads to suffering. Now, suffering is a perfectly natural part of getting a neural network to work well, but it can be mitigated by being thorough, defensive, paranoid, and obsessed with visualizations of basically every possible thing.

### Overview
- The recipe builds from simple to complex and at every step of the way we make concrete hypotheses about what will happen and then either validate them with an experiment or investigate until we find some issue.
- What we try to prevent very hard is the introduction of a lot of “unverified” complexity at once, which is bound to introduce bugs/misconfigurations that will take forever to find (if ever).

### 1. Become one with the data
Spend a lot of time looking through your dataset. Try to find issues (duplicated examples, corrupted data, data imbalances, or biases).

Get an intuition for what your brain is doing to classify the examples: 
- Are the features very local or do you need global context?
- How much variation is there?
- Does spatial position matter or can you average pool it out?
- How noisy are the labels?

It's also a good idea to visualize the distribution of your dataset and look at any outliers. Outliers are usually indicative of bugs.

### 2. Set up end-to-end training/eval + get baselines







---
Reflections on my work on SeqFormer:
- Made too quick a jump to implement because under time crunch.
- Should have had more visualizations early on (ex. bounding box). Would this have made a difference?

Reflections on PyTorch migration:
- What we try to prevent very hard is the introduction of a lot of “unverified” complexity at once, which is bound to introduce bugs/misconfigurations that will take forever to find (if ever).