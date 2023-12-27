---
tags:
  - flashcards
source: https://medium.com/@sharathhebbar24/text-generation-v-s-text2text-generation-3a2b235ac19b
summary: Text Generation completes an input piece of text. Text2Text will take an input and then respond to it (vs. just predicting the most likely next tokens).
---
Text Generation (aka Causal Language Modeling) uses a decoder only style architecture where it just completes the model input. It operates in a left-to-right fashion and only looks at the preceding tokens.

Text2Text Generation is used to convert one sequence of text into another (ex. translation or question answering).

**Text Generation using GPT-2:**

![[AI-Notes/Hugging Face/text2text-vs.-text-generation-srcs/text2text-vs.-text-generation-20231224155258498.png]]
[Source](https://medium.com/@sharathhebbar24/text-generation-v-s-text2text-generation-3a2b235ac19b)

**Text2Text Generation using [[Research-Papers/Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer|T5]]:**
![[AI-Notes/Hugging Face/text2text-vs.-text-generation-srcs/text2text-vs.-text-generation-20231224155315418.png]]
[Source](https://medium.com/@sharathhebbar24/text-generation-v-s-text2text-generation-3a2b235ac19b)