---
tags: [flashcards]
aliases: [GNMT]
source: [[GNMT - Google's Neural Machine Translation System_ Bridging the Gap between Human and Machine Translation, Yonghui Wu et al., 2016.pdf]]
summary:
---
Paper: [[GNMT - Google's Neural Machine Translation System_ Bridging the Gap between Human and Machine Translation, Yonghui Wu et al., 2016.pdf]]
- Introduces the concept of **beam search with length-normalization**: "Without some form of length-normalization regular beam search will favor shorter results over longer ones on average since a negative log-probability is added at each step, yielding lower (more negative) scores for longer sentences"

# Segmentation Approaches
### Wordpiece Model (WPM)
- Originally created by [[WordPieceModel - Japanese and Korean Voice Search|WordPieceModel]]

![[example-wordpiece-sequence.png]]
- In the above example, the word `“Jet”` is broken into two wordpieces `“_J”` and `“et”`, and the word `“feud”` is broken into two wordpieces `“_fe”` and `“ud”`. The other words remain as single wordpieces. `“_”` is a special character added to mark the beginning of a word.
- The wordpiece model is generated using a data-driven approach to maximize the language-model likelihood of the training data, given an evolving word definition