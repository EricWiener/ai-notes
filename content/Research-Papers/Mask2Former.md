---
tags: [flashcards]
source: [[Mask2Former_ Masked-attention Mask Transformer for Universal Image Segmentation, Bowen Cheng et al., 2021.pdf]]
summary: builds on the meta architecture introduced in [[MaskFormer]]
---

> [!NOTE] Mask2Former builds on the meta architecture introduced in [[MaskFormer]] and adds:
> - Masked attention in the Transformer decoder which restricts attention to localized features centered around predicted segments.
> - Uses multi-scale high resolution features.
> - Propose optimization improvements.


### Analysis from [[Mask Dino]]
Mask2Former improves MaskFormer by introducing masked-attention to Transformer. Mask2Former has a similar architecture as DETR to probe image features with learnable queries but differs in using a different segmentation branch and some specialized designs for mask prediction. However, while Mask2Former shows a great success in unifying all segmen- tation tasks, it leaves object detection untouched and our empirical study shows that its specialized architecture de- sign is not suitable for predicting boxes.

![[Mask Dino#Why can't Mask2Former perform detection well?]]