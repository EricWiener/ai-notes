---
tags: [flashcards]
source: https://arxiv.org/abs/2201.12086
summary:
---

[YouTube Overview](https://youtu.be/X2k7n4FuI7c)

Most existing [[Vision-and-Language Pretraining|Vision-Language Pre-training]] models only do well in either understanding-based tasks or generation-based tasks. Additionally, performance improvements have mostly come from scaling up the dataset with noisy image-text pairs collected from the web which is a poor source of supervision.

BLIP can handle both vision-language understanding and generation tasks. Additionally, it makes better use of noisy web data by bootstrapping the captions where a captioner generates synthetic captions and a filter removes the noisy ones.

> [!QUESTION] What does bootstrapping captions mean?

# Related Work
### [[CLIP Learning Transferable Visual Models From Natural Language Supervision|CLIP]] vs. BLIP
![[blip-zero-shot-image-text-retrieval.png]]
BLIP does much better than CLIP with fewer pre-training images. 