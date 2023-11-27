---
tags:
  - flashcards
source: https://arxiv.org/abs/2203.01305
summary: improves on [[End-to-End Object Detection with Transformers|DETR]] by adding ground truth labels and boxes with noises into the Transformer decoder layers to help stabilize bipartite matching during training.
---

[Helpful YouTube Video](https://youtu.be/AqTBqxS0Uww)

They blame the slow convergence of [[DETR|DETR]] on the unstable bipartite matching to decide which predicted boxes correspond to which ground truth boxes during training. After updating weights of the model, a query might be matched with different objects in different epochs. This makes training very unstable and training DETR takes 500 epochs on COCO detection dataset vs. [[Faster R-CNN]] which only takes 12 epochs.

Bipartite matching is also not differentiable, so you can't end-to-end optimize the model.

![[dn-detr-performance-graph.png]]
DN-Deformable DETR converges faster and gets better performance than both [[Deformable DETR]] and [[DETR|DETR]]. They get the best result among all detection algorithms in the 12-epoch setting.

DN-DETR builds ontop of [[DAB-DETR]] which uses 4D anchor boxes as queries. They add noised GT bounding boxes as noised queries together with the learnable anchor queries into the Transformer decoders. Both kinds of queries have the same input format $(x, y, w, h)$ and can be fed into the Transformer decoders together.

For the noised queries, a denoising task is used to reconstruct the corresponding GT boxes (using a reconstruction loss). The learnable anchor queries use the same training loss and bipartite matching as in vanilla DETR. 

Additionally, the paper uses [[Deformable Attention]] to improve results and convergence speed. They also show the denoising training can improve segmentation models such as [[Mask2Former]] and [[Mask Dino]].

### Why is DETR slow?
A small change in the cost matrix may cause an enormous change in the matching result of the [[Hungarian Algorithm]], which will further lead to inconsistent optimization goals for decoder queries.

![[detr-architecture-diagram.png|800]]
The authors of the paper view the training process of DETR-like models as two stages: learning "good anchors" (aka object queries) and learning relative offsets of the bounding boxes (predicted by [[DETR#Prediction feed-forward network (FFN)|a FFN]] on the output of the decoder). 

Decoder queries are responsible for learning anchors. The inconsistent update of anchors can make it difficult to learn relative offsets.

The below video shows how the objects the queries are matched to can change before/after a parameter update.
![[DETR After Parameter Update.mp4]]

# Denoising task
They use [[DAB-DETR]] as the basic detection architecture and replace the decoder embedding part (4D anchor boxes) to support the label 
- The denoising loss is an easier auxiliary task that can accelerate training and boost performance.
- The target of the noised boxes is to reconstruct the original box. This allows bypassing bipartite graph matching and learning to directly approximate ground truth boxes.
- Each decoder query is interpeted as a 4D anchor box. A noised quiery is considered a "good anchor" which has a corresponding ground-truth nearby. The denoising training therefore needs to predict the original bounding box which avoids the ambiguity caused by the Hungarian matching (and avoids the query switching to different matched boxes).

### Making query search more local
DN-DETR can help detection by reducing the distance between anchors and the corresponding targets. The below figure shows that the distance between the initial anchors (positional queries) and target bounding boxes.
![[dab-detr-vs-dn-detr-anchor-target-distance.png]]
Denoising training trains the model to reconstruct boxes from the noised ones that are close to the ground truth boxes. This makes the model search locally for prediction which makes each query ==focus on local regions== and prevents prediction conflicts between queries.
<!--SR:!2025-10-21,697,275-->

A visualization showing the vectors starting at an anchor to the corresponding ground-truth box is shown below. The shortened distances between anchors and targets makes the training process easier and therefore converge faster.

![[screenshot-2023-01-13_15-57-07.png]]
# Architecture
![[dn-detr-architecture-overview.mp4]]

![[dn-detr-architecture-annotated|700]]
They base their architecture on [[DAB-DETR]] which formulates the decoder queries as box coordinates. The only change they make is modifying the decoder embedding to be a class label embedding to make it easier to do label denoising.

They modify the decoder queries to be either:
- Learnable anchors (like [[DETR]]) whose corresponding decoder outputs are matched to ground-truth box-label pairs with bipartite matching.
- Noised ground-truth box-label pairs whose corresponding outputs attempt to reconstruct the GT objects.

To increase denoising efficiency, they ==use multiple versions of noised GT objects== in the denoising part. Additionally, they  use an attention mask to prevent information leaking.
<!--SR:!2024-03-28,292,277-->

### DETR Architecture
The architecture contains a Transformer encoder and a Transformer decoder. On the encoder side, the image features are extracted with a CNN backbone and then fed into the Transformer encoder with positional encodings to attain refined image features. On the decoder side, queries are fed into the decoder to search for objects through cross-attention. [[DN-DETR]].

### DAB-DETR
Each query is setup as a 4D anchor coordinate of the form $(x, y, w, h$) where $x, y$ are the center coordinates and $w, h$ are the corresponding width and height of each box. The anchor coordinates are dynamically updated by each layer of the decoder.

The output of each decoder layer contains a tuple $(\Delta x, \Delta y, \Delta w, \Delta h)$ and the anchor is updated to $(x+\Delta x, y+\Delta y, w+\Delta w, h+\Delta h)$.

### Denoising
Denoising is used to improve model training. It is removed during inference (regular bipartite matching is used).

**Noising boxes**:
They add noise to boxes by **center shifting** and **box scaling**.

**Label Flipping**:
They randomly flip some GT labels to other labels. This forces the model to predict the GT labels according to the noised boxes to better capture the label-box relationship.

### Attention Mask
They use masked attention to make sure the denoising training won't compromise performance. They use multiple **denoising groups** during training where each denoising group contains ==$M$ queries (where $M$ is the number of GT objects in the image) that have a similar noise applied to them==.
<!--SR:!2024-12-29,503,310-->

There are two types of potential information leakages using noised ground truth objects:
??
1. One is that the matching part may see the noised GT objects and easily predict GT objects. 
2. The other is that one noised version of a GT object may see another version.
<!--SR:!2024-04-15,309,290-->

Note that whether the denoising part can see the matching part of not will not influence the performance since the queries of the matching part are learned queries that contain no information about the GT objects (the reverse is not true though - you don't want the matching part to see the GT objects).

### Label Embedding
The decoder embedding is specified as a label embedding to support both box denoising and label denoising. They add an unknown class embedding and an indicator to label whether a query belongs to a denoising part (1) or (0) otherwise.

> [!QUESTION] What is the point of the unknown class embedding?

# Applying Denoising to other Models
### [[Faster R-CNN]]
Faster R-CNN does not have bipartite matching but it does have label assignment controlled by an [[IoU]] threshold. Denoising training can serve as a shortcut to help learn bounding box regression without label assignment.

### [[Mask2Former]]
In Mask2Former, each decoder layer predicts segmentation masks which are passed to the subsequent decoder layer as the attention mask to pool features.

They add noise to the GT masks and feed them to the decoder as attention masks. The objective of these noised masks is to directly predict the GT mask which bypasses the bipartite match and serves as a shortcut to directly learn mask refinement.

The GT mask is noised by shifting the whole mask on the x-axis and y-axis by a random value (similar to center shifting boxes). They don't change the shape or size of the mask.

### [[Mask Dino]]
Mask DINO already has both box denoising and mask denoising. [[DN-DETR]] only keeps the box denoising part to make a fair comparison.

# Results
- The results show that DN-DETR achieves the best results among single-scale models with all four commonly used backbones.
- While denoising training introduces a minor training cost increase, it only needs about half the number of training epochs (25 epochs). This results in a significant net speed-up.
- Adding more denoising groups improves performance, but the performance improvement becomes marginal as the number of denoising groups increases. Therefore, in our experiment, our default setting uses 5 denoising groups, but more denoising groups can further boost performance as well as faster convergence.
- Both center shifting and box scaling improve performance. But when the noise is too large, the performance drops.
- [[DN-DETR]] is only a training method and also com- patible with other methods, for example, deformable attention , semantic-alignment, and query selection, etc.