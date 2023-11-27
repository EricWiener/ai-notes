---
tags: [flashcards]
source: 
summary: panoptic segmentation with a mask transformer.
---

[[DeepLab]], [[DeepLabv3]], and [[DeepLabv3+]] all do semantic segmentation. MaX-DeepLab does end-to-end [[Panoptic Segmentation]]. 

[YouTube Video](https://youtu.be/ir0Avw92Jv0)

Previous approaches for panoptic segmentation (ex. Panoptic-FPN) required predicting boxes, instance segmentation, and semantic segmentation individually and then merging components. MaX-DeepLab instead uses a transformer to directly predict segmentation masks.

Max-Deeplab directly predicts object categories and masks through a dual-path transformer regard-less of the category being things or stuff. [[Panoptic SegFormer]].

Max-DeepLab removes the dependence on box predictions for panoptic segmentation with conditional convolutions. However, in addition to the main mask classification losses it requires multiple auxiliary losses (i.e., instance discrimination loss, mask-ID cross entropy loss, and the standard per-pixel classification loss). [[MaskFormer]].

### Related Work
![[max-deeplab-fig2-comp-of-state-of-art.png]]
(a) Our end-to-end MaX-DeepLab correctly segments a dog sitting on a chair. (b) Axial-DeepLab relies on a surrogate sub-task of regressing object center offsets. It fails because the centers of the dog and the chair are close to each other. (c) DetectoRS classifies object bounding boxes, instead of masks, as a surrogate sub-task. It filters out the chair mask because the chair bounding box has a low confidence.