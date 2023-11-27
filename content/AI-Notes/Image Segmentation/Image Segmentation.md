---
tags: [flashcards, eecs442, eecs498-dl4cv]
source:
summary:
---
[[lec11-scene.pdf]]
[[498_FA2019_lecture16.pdf]]

[[Semantic Segmentation]] assigns class labels to pixels in an image (with no distinction between different entity instances). Semantic segmentation works well for object categories that can't be seperated into instances (e.g. sky, grass, water, trees). Object detection can only handle 

Instance segmentation is to predict a mask and its corresponding category for each object instance. Semantic segmentation requires to classify each pixel including the background into different semantic categories. Panoptic segmentation unifies the instance and semantic segmentation tasks and predicts a mask for each object instance or background segment.

# **Things and stuff**:
In computer vision object categories can be divided into two main classes: **things and stuff**. Things are object categories that can be separated into individual instances (cats, cars, persons). **Stuff** are objects that cannot be separated (grass, sky, trees). 

**Handling things and stuff**:
- [[Object Detection]] can only handle things (individual entities).
- [[Semantic Segmentation]] can handle things and stuff, but merges instances.
- [[Instance Segmentation]] can handle things and stuff (individual entities).
- [[Panoptic Segmentation]]: like instance segmenation, but adds back in the concept of "things". You have per-entity masks for things and just masks for stuff.

Object detection gives bounding boxes for things. Semantic segmentation gives per-pixel labels, but merges instances. **Instance segmentation** detects all objects in the image and identifies the pixels that belong to each object (only things - not stuff).

# Other segmentation tasks:
- [[Keypoint Estimation]]: represent the pose of a human by locating a set of keypoints.
- [[Predicting Depth]]
- [[3D Shape Prediction]]: Predict a 3D triangle mesh per region. This can be done with Mask R-CNN + Mesh Head (called Mesh R-CNN).

> [!note] The general idea here is that you can add per-region "heads" to Faster/Mask R-CNN for whatever your task is
> 
