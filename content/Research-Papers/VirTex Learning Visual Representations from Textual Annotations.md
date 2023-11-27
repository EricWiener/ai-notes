---
aliases: [VirTex]
tags: [flashcards, pass-three]
---
[Professor Johnson's Talk](https://www.youtube.com/watch?v=L-Fx-7WqPPs)
Code and docs are all online: https://github.com/kdexd/virtex

[[VirTex Learning Visual Representations from Textual Annotations.pdf|My presentation on VirTex for ML Paper Party]]
[[VirTex Learning Visual Representations from Textual Annotations (Annotated).pdf|Presentation with speaker notes]]

# Quick Summary
![[screenshot-2022-02-25_08-56-06.png]]
      
- Jointly train a ==ConvNet and Transformers== using ==image-caption== pairs (from [[COCO]]), for the task of image captioning (top). Then, we transfer the learned ConvNet to several ==downstream vision tasks==, for example object detection (bottom).
- Uses captions because they have ==high semantic density== (capture a lot of meaning).
- Learn visual features directly from language annotations. Then transfer CNN to visual tasks. No other approaches put the vision task downstream from vision + language pre-training.
<!--SR:!2025-04-27,639,330!2026-12-13,1363,350!2024-07-07,280,332!2024-07-08,281,332-->

# Current approaches in CV for training for scale
**Unsupervised Pretraining in NLP**
- With BERT they perform unsupervised masked language modeling.
- This requires no human labels. Can just download as much data as you want from the internet.

**Supervised Pretraining in Vision**
- Currently relies on images being labelled by humans.
- Ex: you train a ResNet-50 backbone on ImageNet. Then, you can transfer the backbone to use it for object detection.

[[Webly-Supervised Pretraining]]: 
![[Webly-Supervised Pretraining#Problems]]
- VirTex: uses fewer images with higher-quality annotations.  Captions have higher semantic density.

[[Self-supervised Learning]]:
![[Self-supervised Learning#Problems]]
- VirTex: "we leverage textual annotations for semantic understanding. Unlike these methods, our approach can leverage additional metadata such as text, when scaled to internet images”.

[[Vision-and-Language Pretraining]]:
![[Vision-and-Language Pretraining#Problems]]
- VirTex: pretrain via image captioning, and put vision tasks downstream from vision-and-language pretraining.

![[comparison-of-all-vision-pre-training.png]]

# Method
See annotated slides

### Why train with image + text:
- A caption tells us a lot of information about an image. It is ==semantically dense==.
- Hopefully having semantically dense annotations will be more data efficient than other methods of learning features.
<!--SR:!2025-08-22,955,310-->

### Tricky Optimization Details
- Used lookahead optimizer
- Use a different learning rate for CNN and the Transformer
- Train for a long time (trained from scratch on [[COCO]]).
- Sped up with mixed precision training.
- Training time was 3 days with 8 2080 TIs.


# Results
See annotated slides

# Summary
- Learning visual features directly from language annotations gives high-quality features from fewer images
- Training CNN + Transformer on COCO Captions matches or exceeds ImageNet pretraing on several standard downstream tasks and datasets.
    - Classification on VOC
    - Object Detection on VOC
    - Instance Segmentation on COCO
    - Few-shot Instance Segmentation on LVIS
- Can drop in model with one line of code (using Torch Hub)

**Future work**:
- Scale to larger web-scale data
- Use the textual head for downstream tasks.

# First Draft Notes
### Results
- Decent results for Image Captioning (SOTA in 2016). Doing reasonably, but not amazing. All models are above human performance at this point. The captions look reasonable and attention maps look good. This was pre-text task.
- The model performs better with less images than other models. Using captions is better than using classification labels. Training on 118K COCO images outperforms 1.2 million ImageNet images (x10 fewer images).
- Better off having one caption for one image than more caption for fewer images.

### Ablation Studies
- Used bi-directional captioning, one directional captioning, token classification (multi-label classification over the tokens), multi-label classification (using object detection labels from COCO), instance segmentation training (train Mask-RCNN on COCO and then pull out the backbone and transfer it to different tasks).
- Using bi-directional captioning (makes use of the structure of language) performed the best.
- Using larger visual backbones showed benefits.
- Wider transformers improve downstream task performance. The transformer gets thrown away after pre-training (only CNN is kept). One idea is that having the larger transformer takes up the brunt of the text processing and lets the CNN focus on the image processing. Deeper transformers also improve performance.
- Also looked at Object Detection.
- Fine-grained, few-shot detection on [[LVIS]] dataset (1230 categories with few examples). VirTex outperforms all other models on LVIS. Intuition: language annotations might do better when trying to recognize long-tail of objects (less frequent).

### Questions
- What is MoCo and PIRL?

## Results
### Pascal VOC Linear Classification
- Pre-train on COCO captions. Transfer the CNN and train a linear classifier on Pascal VOC.
- Do with varying amounts of annotated images to analyze data efficiency.
- ImageNet-Sup: ImageNet supervised pre-training. More images -> better performance
- MOCO and PIRL are other self-supervised approaches. Almost as good as supervised pre-training.
- VirTex performs better than ImageNet pre-training for multiple scales of images. Even the 1 caption model does well (not just 5 caption model).
- MOCO-COCO is MOCO model pre-trained on COCO (vs. ImageNet)
- Conclusion: using captions is more data efficient than using labels.
- 118K COCO images outperforms 1.2M ImageNet images (x10 fewer images)
- Better off having 1 caption for more images than more captions for fewer images (if you have a budget).

### ImageNet Linear Classification
- Very unfair to their model since the baseline of ImageNet supervised pre-training is the same downstream task the model is evaluated on.
- Training on COCO captions is very different.
- Performs almost as well for the same number of images compared to directly learning with the same task as the downstream task.

### Pretext Task Ablation
- Tried bidirectional captioning
- Forward captioning: left->right
- Token classification: model caption as tokens where each token is a label.
- Multi-label classification: uses object instance annotations. Try to classify which objects are present using object detection labels.
- Instance segmentation: train Mask-RCNN from scratch on COCO instance segmentation and then pull out the backbone and transfer to downstream tasks. How does this use text???

### Architecture Ablations
- Larger visual backbones improves donsream performance
- Wider transformers improve downstream task performance. The transformer gets thrown away after pre-training (only CNN is kept). One idea is that having the larger transformer takes up the brunt of the linguistic part of the model and lets the CNN focus on the image processing. Deeper transformers also improve performance.

### Object Detection
- Used Faster RCNN backbone. VirTex model performs about the same with x10 fewer images.

### Instance Segmentation
-  Similar results

### LVIS
- Recognize 1230 entry-level categories.
- Long-tail of categories.
- Few examples for some of the categories in the training data.
- For this task, VirTex outperforms all other methods.
- Intuition: language-based annotations might do better when trying to recognize long-tail of objects.

