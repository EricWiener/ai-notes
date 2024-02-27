---
tags:
  - flashcards
aliases:
  - WordPieceModel
summary: an iterative algorithm for creating an embedding for a vocabulary
source: https://ieeexplore.ieee.org/document/6289079
---

[[WordPieceModel - Japanese and Korean Voice Search.pdf]]

We developed a technique (WordPieceModel) to learn word units from large amounts of text automatically and incrementally by running a greedy algorithm as described below. This gives us a user-specified number of word units (we often use 200k) which are chosen in a greedy way without focusing on semantics maximize likelihood on the language model training data (the same metric that we use during decoding).

> [!question] What does "maximize likelihood on the language model training data" mean?
> Does this mean you train a full model on the training dataset? I don't get what this does.

### Algorithm
The algorithm to find the automatically learned word inventory efficiently works in summary as follows:
1. Initialize the word unit inventory with the basic Unicode characters and including all ASCII, ending up with about 22,000 total for Japanese and 11000 for Korean.
2. Build a language model on the training data using the inventory from 1.
3. Generate a new word unit by combining two units out of the current word inventory to increment the word unit inventory by one. Choose the new word unit out of all possible ones that increases the likelihood on the training data the most when added to the model.
4. Goto 2 until a predefined limit of word units is reached or the likelihood increase falls below a certain threshold.

Training of the segmenter is a computationally expensive procedure if done brute-force as for each iteration (addition of a new word piece unit by combining two existing ones), as all possible pair combinations need to be tested and a new language model needs to be built. This would be of computational complexity $O(K^2)$ **per iteration** with $K$ being the number of current word units.