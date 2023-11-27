---
tags: [flashcards]
source: [[MaskFormer_ Per-Pixel Classification is Not All You Need for Semantic Segmentation, Bowen Cheng et al., 2021.pdf]]
summary: convert a semantic segmentation model to predict instance masks.
---

> [!NOTE] Introduces a meta framework to convert a semantic segmentation model to work for instance segmentation.
> MaskFormer predicts $N$ masks (where $N$ is not necessarily equal to the number of categories, $K$). If doing instance segmentation, you can use the $N$ masks as $N$ individual instances (which may or may not have the same category $K$). If doing semantic segmentation, you can merge all masks that have the same category label. If doing panoptic segmentation, you can merge all "stuff" masks and leave the "thing" masks seperate.

The new model solves both semantic and instance-level segmentation tasks in a unified manner: no changes to the model, losses, and training procedure are required. Specifically, for semantic and panoptic segmentation tasks alike, MaskFormer is supervised with the same per-pixel binary mask loss and a single classification loss per mask.

# Related Work
**Per-pixel classification:**
Per-pixel classification became the dominant approach for semantic segmentation after [[Fully Convolutional Networks for Semantic Segmentation|FCNs]]. 

Modern semantic segmentation models focus on aggregating long-range context in the final feature map:
- [[DeepLab#Atrous Spatial Pyramid Pooling (ASPP)]]  uses atrous convolutions with different atrous rates.
- [[Segmenter - Transformer for Semantic Segmentation|Segmenter]] replaces the traditional convolutional backbones with [[ViT An Image is Worth 16x16 Words Transformers for Image Recognition at Scale|ViT]] to capture long-range context starting from the very first layer.

**Mask classification:**
Mask classification is commonly used for instance-level segmentation tasks. Unlike per-pixel classification which predicts one label for every pixel, you can have a dynamic number of masks.

[[Mask R-CNN]] uses a global classifier to classify mask proposals for instance segmentation.

[[DETR]] uses a [[Transformer]] design to handle things and stuff segmentation for [[Panoptic Segmentation]], but their mask classification requires predicting bounding boxes. MaskFormer claims this may limit their usefulness for semantic segmentation, but [[Mask Dino]] saw an improvement from supervising bounding box predictions for "things" (but not "stuff").

# From Per-Pixel to Mask Classification
For per-pixel classification, a segmentation model aims to predict the probability distribution over all possible $K$ categories for every pixel of an $H \times W$ image. For every pixel you predict a label from the set of $K$ categories (ground truth labels are of form $y^{\mathrm{gt}}=\left\{y_i^{\mathrm{g}} \mid y_i^{\mathrm{gt}} \in\{1, \ldots, K\}\right\}_{i=1}^{H \cdot W}$ ).

Mask classification splits the segmentation task into two parts:
1. Partition the image into $N$ regions ($N$ does not need to equal $K$) represented with binary masks.
2. Assign a label to each of the $N$ regions from the set of $K+1$ categories (you have an additional $\varnothing$ category predicted for masks that don't correspond to any of the $K$ categories).

Mask classification allows multiple mask predictions with the same associated class (from the $K$ categories) which makes it applicable to both semantic and instance segmentation.

To train a mask classification model you need a way to match between the set of predicted masks and the set of ground truth masks. In the special case you are doing semantic segmentation, a fixed matching is possible if the number of predictions $N$ is the same as the number of category labels $K$.

Unlike [[DETR]] that uses bounding boxes to compute the assignment costs between the predicted regions and ground truth regions, MaskFormer uses the masks.

# Results
While MaskFormer performs on par with per-pixel classification models for Cityscapes, which has a few diverse classes, the new model demonstrates superior performance for datasets with larger vocabulary.
