---
tags: [flashcards]
source: https://arxiv.org/abs/1905.04804
summary: this paper introduces the VIS task and modifies Mask-RCNN to support tracking (new model named MaskTrack R-CNN).
---

### Overview
[MaskTrack R-CNN code](https://github.com/youtubevos/MaskTrackRCNN) based on mmdetection. MaskTrack R-CNN extends the Mask R-CNN with a tracking branch and external memory that saves the features of instances across multiple frames.

### Tracking Branch
They add a fourth branch to Mask R-CNN (classifiation, bounding box regression, binary segmentation, and tracking).

If there are $N$ already identified instances already identified in the previous frames, a new candidate box can either be assigned to one of the $N$ identities or assigned as a new instance (this is represented as multiclass classification with $N + 1$ classes).

The tracking branch has two fully connected layers which project the feature maps extracted by [[RoI Align]] into new features. The features for previously identified instances are stored in memory to avoid recomputing them. New instances are added to the memory and existing instances are updated based on the most recent matching.