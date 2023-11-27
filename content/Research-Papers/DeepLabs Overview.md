---
tags: [flashcards]
source:
summary: an overview of the different DeepLab papers.
---

# Semantic Segmentation
### [[DeepLab]](v1 = 2016, v2 = 2017)
Initial version (v1 = VGG, v2 = ResNet)

### [[DeepLabv3]] (2017)
Improved ASPP + no CRF

### [[DeepLabv3+]] (2018)
Uses Xception backbone and decoder

# Panoptic Segmentation
### [[Panoptic-DeepLab]] (2020)
Supports panoptic segmentation.

### [[ViP-DeepLab]] (2020)
We extend Panoptic-DeepLab to perform center regression for two consecutive frames with respect to only the object centers that appear in the first frame. During inference, this offset prediction allows ViP-DeepLab to group all the pixels in the two frames to the same object that appears in the first frame. New instances emerge if they are not grouped to the previously detected instances. This inference process continues for every two consecutive frames (with one overlapping frame) in a video sequence, stitching panoptic predictions together to form predictions with temporally consistent instance IDs.

### [[Axial-DeepLab]] (2020)
Improves on Panoptic-DeepLab.

### [[MaX-DeepLab]] (2020)
Improves on Axial-DeepLab

### [CMT-DeepLab](https://ar5iv.labs.arxiv.org/html/2206.08948) (2022)
Improves on MaX-DeepLab but is soon replaced by kMaX-DeepLab so no source code is released.

### [kMaX-DeepLab](https://github.com/google-research/deeplab2/blob/main/g3doc/projects/kmax_deeplab.md) (2022)
[kMaX-DeepLab](https://arxiv.org/pdf/2207.04044.pdf) is an end-to-end method for general segmentation tasks. Built upon [MaX-DeepLab](https://arxiv.org/pdf/2012.00759.pdf) [1] and [CMT-DeepLab](https://arxiv.org/pdf/2206.08948.pdf) [2], kMaX-DeepLab proposes a novel view to regard the mask transformer [1] as a process of iteratively performing cluster-assignment and cluster-update steps.

1.  CMT-DeepLab v.s. kMaX-DeepLab: Both methods regards the cross-attention as a clustering process from a high-level and further reformulate the mask transformer as a process of interactively clustering assignment and update (I.e., Fig.2 in [CMT-DeepLab's paper](https://arxiv.org/pdf/2206.08948.pdf)). Though both works notice the clustering illustration, CMT-DeepLab mainly focuses on improving original cross-attention, with a complementary clustering term for update. In kMaX-DeepLab, however, we notice that replacing original cross-attention is doable, and we can achieve a much better performance simply by making the whole process to be even more like k-means clustering. So in short, CMT-DeepLab and kMaX-DeepLab share a similar motivation, but kMaX is more like k-means and much simpler, with a better performance.
    
2.  I think both CMT-DeepLab and kMaX-DeepLab density the attention compared to original cross-attention, as both methods try to assign pixels to objects instead of objects to pixels. (I.e., both methods are expected to involve more pixels to update the cluster centers/object queries).
    
3.  As kMaX-DeepLab is not only an improved but also a much simplified method over CMT-DeepLab, we currently have no plan for CMT-DeepLab.