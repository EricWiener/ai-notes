---
aliases: [weakly supervised learning]
tags: [flashcards]
---
**As described by [[VirTex Learning Visual Representations from Textual Annotations|VirTex]]**:
- Learn visual features on ==**noisy** (image, label) pairs== downloaded from the web. 
- Large quantities of images with noisy labels.
    - [[JFT-300M]] is a Google dataset of ~300M images annotated with labels from ~18k categories.
    - Instagram pre-training. Train on ~3.5B Instagram images with ~17K user-provided hashtags.
<!--SR:!2024-01-25,472,310-->

### Problems:
- Very ==data inefficient==. Need to download a lot of images to get good results. 
    - 6.6M ImageNet images with with classification labels gives similar performance as 940M Instagram images with hashtags (1 ImageNet image = 142 Instagram images). [[Exploring the Limits of Weakly Supervised Pretraining]]
<!--SR:!2026-12-15,1365,350-->