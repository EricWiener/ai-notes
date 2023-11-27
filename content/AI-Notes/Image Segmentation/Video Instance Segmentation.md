---
tags: [flashcards]
aliases: [VIS]
summary: simultaneously perform detection, classification, segmentation, and tracking of object instances in videos.
---

> [!NOTE] How is video instance segmentation different than video object segmentation or video object detection?
> Video object segmentation aims at segmenting and tracking objects in videos, but does not require recognition of object categories. Video object detection aims at detecting and tracking objects, but does not deal with object segmentation or recognition of object categories.

# Paper Overview
### [[MaskTrack R-CNN]]
The paper that published this model also was the first to introduce the Video Instance Segmentation task.
![[MaskTrack R-CNN#Overview]]


### [[End-to-End Video Instance Segmentation with Transformers|VisTR]]
![[End-to-End Video Instance Segmentation with Transformers#Overview]]

### [[IFC]]
IFC improves the performance and efficiency of VisTR by building communications between frames in a transformer encoder.

### [[MinVIS]]

### [[MS-STS]]

### [[TeViT]]

### [[SeqFormer]]
Uses 

### [[CrossVIS]]
Based on detectron2. Seems to convert to TRT, but gets worse performance than SeqFormer. 

CrossVIS proposes a learning scheme that uses the instance feature in the current frame to pixel-wisely localize the same instance in other frames.