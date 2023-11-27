[Project page](https://jerryxu.net/GroupViT/)

> [!note]
> Without using any mask annotations, GroupVit groups the image into segments and outputs semantic segmentation map.
![[Zero-shot learning]]

We propose GroupViT to zero-shot transfer to semantic segmentation with text supervision only.
- This means they will have the model perform semantic segmentation on classes it hasn't seen images of before given only textual data.

> [!note]
> This could be used at Zoox by having drivers dictate whats going on while they drive to try to get textual data that corresponds to the scene.

![[Pascal VOC 2012]]

# Introduction
- Leverage [[ViT An Image is Worth 16x16 Words Transformers for Image Recognition at Scale|ViT]]

# Training
![[screenshot-2022-02-23_14-04-26.png]]
- Inputs are image-text pairs crawled from the web.
- Train GroupViT jointly with a Text Encoder via [[Contrastive Learning]]
- Groups emerge without any annotation
- Extract image segments from GroupViT
- Assign a label to each segment according to its similarity to the text embedding of class labels.
