---
tags: [flashcards]
source: https://arxiv.org/abs/2206.02777
summary:
---

[Code](https://github.com/IDEA- Research/MaskDINO)

Mask DINO extends DINO ([[DINO|DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection]]) by adding a mask prediction branch which supports [[Instance Segmentation]], [[Semantic Segmentation]], and [[Panoptic Segmentation]].

It uses the query embeddings from DINO to dot-product a high-resolution pixel embedding map to predict a set of binary masks.

Mask DINO establishes the best results to date on instance segmentation (54.5 AP on COCO), panoptic segmentation (59.4 PQ on COCO), and semantic segmentation (60.8 mIoU on ADE20K).

Like DINO, Mask DINO incorporates both box denoising (similar to [[DN-DETR]]) and mask denoising.

**Two questions this paper aims to answer:**
1) why cannot detection and segmentation tasks help each other in Transformer-based models?
2) is it possible to develop a unified architecture to replace specialized ones?

# Related Work
Convolution-based algorithms have developed specialized architectures like [[Faster R-CNN]] for object detection, [[Mask R-CNN]] for instance segmentation, and [[Fully Convolutional Networks for Semantic Segmentation|FCN]] for semantic segmentation. These algorithms are tailored for specialized tasks and lack the generalization ability to address other tasks. Task unification not only helps simplify algorithm development but also brings in performance improvement in multiple tasks.

[[DETR]] addresses both object detection and panoptic segmentation, but its segmentation performance is still inferior to convolution-based approaches and the paper only includes the segmentation head to show it is feasible to extend the model for segmentation.

[[DINO]] is able to achieve SOTA performance for object detection by building on [[DAB-DETR]], [[DN-DETR]], and [[Deformable DETR]].

[[MaskFormer]] and [[Mask2Former]] propose to unify different image segmentation tasks using query-based Transformer architectures to perform mask classification. Such methods have achieved remarkable performance improvement on multiple segmentation tasks

### Why can't [[Mask2Former]] perform detection well?
**It doesn't base positional decoder queries on encoder features**
It's decoder queries follow the design of DETR without being able to utilize information from the encoder to get better positional priors as in [[DAB-DETR]]. It's content queries are semantically aligned with the features from the Transformer encoder, but the positional queries are just learned vectors (static at inference time) as in vanilla [[DETR]].

If you remove the mask branch of [[Mask2Former]] it reduces to a variant of [[DETR]] whose performance is worse than the papers that improve on it ([[Deformable DETR]], [[DAB-DETR]], [[DN-DETR]], etc.).

**It uses high resolution masked attention which adds hard-constraints**
The attention masks predicted from a previous layer are high resolution and used as hard-constraints for attention computation. They are neither efficient nor flexible for box prediction.

**It can't perform iterative box refinement**
It doesn't have an explicit formulation of the queries as being boxes, so you can't predict a box for each layer and then embed the box back to the original embedding space.

**It doesn't use multi-scale features from the encoder**
This makes it hard to learn about large regions. Mask2Former is designed to operate on a pixel scale so it's okay for it to just use high resolution inputs when predicting masks.

### Why can't [[DETR]]/[[DINO]] perform segmentation well?
Adding DETR's segmentation head or Mask2Former's segmentation head result in inferior performance to Mask2Former.

**DETR's segmentation head is not optimal**
DETR lets each query embedding dot-product with the lowest resolution feature map to compute attention maps and then upsamples them to get the mask predictions. This design lacks an interaction between queries and higher resolution feature maps from the backbone.

**Features in detection models are not aligned with segmentation**
For example, DINO inherits many designs from like query formulation, denoising training, and query selection. However, these components are designed to strengthen region-level representation for detection, which is not optimal for segmentation.

# Architecture
![[mask-dino-architecture.png]]
The framework of Mask DINO, which is based on DINO (the blue-shaded part) with extensions (the red part) for segmentation tasks. ’QS’ and ’DN’ are short for query selection and denoising training, respectively.


### Mask Prediction Branch
Mask DINO extends DINO with a mask prediction branch in parallel to DINO's box prediction branch.

For image segmentation, they follow a similar approach to [[MaX-DeepLab]] and reuse content query embeddings (outputs of the decoder) from DINO to perform mask classification for all segmentation tasks on a high-resolution pixel embedding map (1/4 of the input image resolution) obtained from the backbone and Transformer encoder features.

The mask branch predicts binary masks by dot-producting each content query embedding with the pixel embedding map.

# Boosting segmentation performance
DINO is a detection model designed for region-level (not pixel-level) predictions so Mask-DINO introduces three new components to boost segmentation performance:
??
- A unified and enhanced query selection: the same top-K tokens are used for both masks and boxes (you don't have seperate tokens for each). Mask DINO initializes both the content and anchor queries based on encoder output whereas DINO only initializes anchor box queries and the content queries are fixed at inference.
- A unified denoising training for masks: use noised GT boxes as the ground truths for masks.
- Hybrid bipartite matching that includes classification, bounding box, and mask loss in the matching cost used by the Hungarian Algorithm to ensure masks and boxes align.
<!--SR:!2024-01-06,96,150-->

### A unified and enhanced query selection

> [!NOTE] Mask DINO initializes both the content and anchor queries whereas DINO only initializes anchor box queries.

**Unified query selection for mask**:
The encoder output features contain lots of useful information that can be used for the content queries of the decoder.

They add a classification, detection (BBOX), and segmentation head that operates independently on each encoder output token. They select the top highest confidence tokens (based on classification head). The predicted boxes and masks will be supervised by the ground truth (predictions will be used in loss calculation at each layer). 

Unified query selection just means the same top-K tokens are used for both masks and boxes (you don't have seperate tokens for each).

**Mask-enhanced anchor box initialization**:
Image segmentation is a pixel-level classification while object detection is a region-level task. Therefore, segmentation is easier to learn in the initial stage of the model.

In the decoder the initial inputs are based on the unified query selection. However, all other layers of the decoder will refine on the predictions from the previous layer. Mask DINO saw performance improvements deriving anchor boxes from the predicted mask vs. using the predicted box. Therefore, they update the anchors for the next decoder layer using an anchor box derived from the predicted mask of the previous layer.

### A unified denoising training for masks
Denoising has improved training for [[DN-DETR]] and [[DINO]], so they use it here as well. They extend denoising to work for masks (not just boxes). They consider masks a more fine-grained representation of boxes, so they use boxes as a noised version of masks and train the model to predict masks given boxes as a denoising task.

The boxes given to the model are randomly noised to make training more efficient.

### Hybrid bipartite matching
Mask DINO predicts boxes and masks with two parallel heads. This makes it so that the outputs from each head could be inconsistent. To address this, they add a mask prediction loss to the **matching cost** used by the Hungarian Algorithm to encourage consistent matching results for one query.

### Decoupled Box Prediction
When doing panoptic segmentation, there is no need to predict boxes for the "stuff" category. They therefore don't calculate box loss and box matching for "stuff" categories. The box loss for "stuff" is set to the mean of "thing" categories.

They saw that using decoupled box prediction increased performance.

# Results
Mask DINO significantly improves all segmentation tasks and achieves the best results on instance (54.5 AP on COCO), panoptic (59.4 PQ on COCO), and semantic (60.8 mIoU on ADE20K) segmentation among models under one billion parameters.

Using more decoder layers will help performance. They hypothesize that multi-task training beconmes more complex and requires more decoders.

Mask DINO shows that detection and segmentation can help each other in query-based models. In particular, Mask DINO enables semantic and panoptic segmentation to benefit from a better visual representation pre-trained on a large-scale detection dataset.

### Comparison to [[Mask2Former]]
Mask DINO outperforms Mask2Former on all three tasks with less training epochs. Additionally, instead of using dense and hard-constrained mask attention, Mask DINO predicts boxes and then uses them in deformable attention to extract query features.

Mask DINO significantly outperforms Mask2Former on detection. This is due to Mask DINO being an extension of DINO (which does object detection) so it can use a DINO model pre-trained on an object detection dataset and then fine tune it. Mask2Former can not due detection and therefore can't be fine tuned on detection datasets.

Mask2Former also predicts the masks of learnable queries as initial region proposals.

Mask2Former saw that concatenating multi-scale features as input to the Transformer decoder layers did not improve the segmentation performance, but Mask DINO saw that using more feature scales in the decoder improves performance.

