---
tags: [flashcards]
aliases: [VLP, Vision-Language Pre-training]
source:
summary:
---

- Learn vision and language features on (image, text) data.
- Most approaches extract regions from images, tokens from text, and then pass set of data into transformer to perform masked language modeling.
- This model is then transferred for other tasks (image captioning, or question/answering).
- The general approach is to:
    - Extract regions from the images
    - Extract tokens from the text
    - Pass the set of (image regions, text tokens) into a transformer that performs some sort of masked lanaguage modeling (inspired by [[BERT]]).
    - They will then use the trained model on different downstream tasks.
The general training pipeline is:
    1. Train CNN for classification on [[ImageNet]]
    2. Adapt (1) as [[Fast R-CNN]] backbone, finetune on [[Visual Genome]] to train an object detector.
    3. Train BERT on a bunch of web text
    4. Combine (2) and (3) into a joint model, fine-tune on [[Conceptual Captions]].
    5. Fine-tune (4) for a vision+language end task (captioning, VQA, retrieval)

### Benefits
- They learn joint vision and language features using a lot of internet data.

### Problems
![[vision-language-pretraining-problem.png]]
- Vision-and-language tasks are ==downstream== from the initial visual representations learned on ImageNet. CNN often froze after (2), so visual features don't depend on text at all. Doesn't learn visual features - relies on supervised vision pretraining.
- Visual features don't ==depend on text== at all.
<!--SR:!2025-03-24,588,310!2030-01-14,2243,350-->