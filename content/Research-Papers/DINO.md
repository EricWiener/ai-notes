---
tags: [flashcards]
aliases: [DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection]
source: https://arxiv.org/abs/2203.03605
summary: DINO improves over previous DETR-like models in performance and efficiency by using a contrastive way for denoising training, a mixed query selection method for anchor initialization, and a look forward twice scheme for box prediction.
---


> [!NOTE] DINO is an object detection model (predicts BBOX).
> It follows on DAB-DETR (formulating decoder queries as 4D anchors), DN-DETR (adding noised GT boxes as directly predicting the correct GT box - skipping bipartite matching), and Deformable DETR (introduces Deformable Attention).
> 
> It also adds constrastive denoising training (adding negative/positive noised GT boxes), a mixed query selection (learnable content query + positional queries dependent on encoder features), and a look forward twice scheme for BBOX prediction (include the next layer's BBOX prediction when calculating loss for current layer).


# Overview
To understand DINO, you will probably first want to review [[DAB-DETR]], [[DN-DETR]], and [[Deformable DETR]] (with an emphasis on [[Deformable Attention]]). The paper uses the following from each of them:
- Following [[DAB-DETR]], they formulate queries in decoder as dynamic anchor boxes and refine them step-by-step across decoder layers.
- Following [[DN-DETR]], they add ground truth labels and boxes with noises into the [[Transformer]] decoder layers to help stabilize bipartite matching during training.
- Use [[Deformable Attention]] for its computational efficiency.

**DETR Refresher**:
- Previous [[DETR|DETR]]-like models are inferior to the improved classical detectors (ex. Faster R-CNN) since these have been studied and optimized for longer.
- As a DETR-like model, DINO contains a backbone, a multi-layer Transformer encoder, a multi-layer Transformer decoder, and multiple prediction heads.
- DETR has issues converging which causes training to take a while.

**New methods proposed**
In addition to combining changes from previous papers, DINO also propose three new methods:
??
- Contrastive denoising training. It adds a negative noised BBOX that the model should learn to avoid predicting.
- Mixed query selection method (learnable content query + positional queries dependent on encoder features). Mask DINO will later switch back to having both content and positional dependent on features.
- Look forward twice scheme to change how parameters are updated.
<!--SR:!2023-11-27,30,130-->

# Related Work
### Classical Object Detectors
- Two-stage models ([[Faster R-CNN]], [[Mask R-CNN]]) use a [[Faster R-CNN#Region Proposal Network|region proposal network]] to propose potential boxes in the first stage which are then refined in parallel in the second stage.
- One-stage models such as [[YOLO]] directly predict offsets relative to predefined anchors.
- Recently convolution-based models (HTC++ and Dyhead) have achieved top performance. The performance of convolution-based models, however, relies on the way they generate anchors. Moreover, they need hand-designed components like [[Non-maximum Suppression|NMS]] to remove duplicate boxes, and hence cannot perform end-to-end optimization.

### [[DETR|DETR]] and its variants
- [[DETR|DETR]] is a Transformer-based end-to-end object detector that doesn't use hand-designed components like anchor design and NMS.
- DETR converges slowly because of the decoder cross-attention (the queries will end up matching to different GT boxes as training progresses). Some lines of work focus on improving training.
- [[Deformable DETR]] predicts 2D anchor points and designs a de-formable attention module that only attends to certain sampling points around a reference point.
- [[DAB-DETR]] further extends 2D anchor points to 4D anchor box coordinates to represent queries and dynamically update boxes in each decoder layer.
- [[DN-DETR]] introduces a denoising training method to speed up DETR training. It feeds noise-added ground-truth labels and boxes into the decoder and trains the model to reconstruct the original ones.

# Model Overview
![[dino-architecture.png]]
The main improvements of DINO are in the Transformer encoder and decoder. The top-K encoder features in the last layer are selected to initialize the positional queries for the Transformer decoder (similar to how this was done in [[Deformable DETR]]). The content queries are kept as learnable parameters. The decoder also contains a Contrastive DeNoising (CDN) part with both positive and negative samples.

- Given an image, you extract multi-scale features with a backbone.
- You then add positional embeddings to the extracted features and pass them to the Transformer encoder.
- You get out a sequence of features from the encoder and use two small parallel heads to predict bounding box scores (binary label as was done in [[Deformable DETR#Two-Stage Deformable DETR]]) and bounding box position.
- You take the features and corresponding bounding boxes from the features with the top-K highest confidence bounding boxes and use the predicted boxes to initialize the anchors as positional queries for the decoder.
- You then pass the initialized anchor boxes as positional queries to the decoder. The content queries are still learnable.
- You then use deformable attention to combine the features of the encoder outputs and update the predicted anchors layer-by-layer.
- The final outputs are formed with the refined anchor boxes and classification results predicted by refined content features.

During training you use an additional denoising (DN) branch that skips bipartite matching for noised boxes and directly tries to predict the original box (or reject the box in the case of the negative samples in contrastive denoising by predicting the class as background).

Additionally during training a look forward twice method is used to pass gradients between adjacent layers.

# Proposed Methods
### Contrastive denoising training
To improve on one-to-one matching between queries and objects in the image (you want one query to correspond to one object), they add contrastive denoising training by adding both positive and negative samples of the same ground truth at the same time. 

After adding two different noises to the same ground truth box, we mark the box with a smaller noise as ==positive and the other as negative==. 
<!--SR:!2024-09-13,460,308-->

The contrastive denoising training helps the model to ==avoid duplicate outputs of the same target== because you have one positive and one negative example. It also helps avoid making predictions for anchors with no objects nearby.
<!--SR:!2024-04-12,138,190-->

[[DN-DETR]] improved training by adding denoising queries so the model would learn to make predictions based on anchors which have GT boxes nearby. However, the model lacks a capability of predicting "no object" for anchors with no object nearby. To address this, the paper adds negative examples and has a head that can predict a binary bounding box score which can be used to reject anchors. 

**Implementation**:
![[contrastive-denoising-training-dino.png]]
[[DN-DETR]] generates noised boxes with noise no larger than a hyperparameter $\lambda$ since the paper wants to reconstruct the GT from moderately noised queries.

DINO modifies this to use two hyperparmaeters: $\lambda_1$ and $\lambda_2$ where $\lambda_1 < \lambda_2$. Positive queries have a nose less than $\lambda_1$ and negative queries have a noise between $\lambda_1$ and $\lambda_2$. Small $\lambda_2$ are usually chosen because hard negative samples closer to GT boxes are more helpful to improve performance (helping get rid of duplicate predictions for the same object).

If an image has $n$ GT boxes, a CDN (contrastive denoising) group will have $2 \times n$ queries with each GT box having a  positive and negative query. Additionally, like [[DN-DETR]], they use multiple noise groups.

**Why does constrastive denoising improve performance:**
??
Because it helps the model learn to select high-quality anchors for predicting bounding boxes when multiple anchors are close to one object. [[DN-DETR]]'s denoising training helped the model to choose nearby anchors via. noised GT boxes being used as queries. CDN improves on this further by teaching the model to reject anchors further from the GT object.
<!--SR:!2025-09-26,725,308-->

### Mixed query selection
[[DAB-DETR]] discussed how queries in DETR are formed by a positional part and a content part. See [[DAB-DETR#Queries in DETR]].

DINO proposes a mixed query selection method, which helps better initialize the queries. They select initial anchor boxes as positional queries from the output of the encoder, similar to [[Deformable DETR]]. However, we leave the content queries learnable (updated via backprop, but not dependent on current input) as earlier models do, encouraging the first decoder layer to focus on the spatial prior.

**Types of query initialization:**
![[types-of-query-initialization-dino.png]]
There are three types of query initialization:
- Static queries where both the positonal and content information remains the same for different images during inference.
- Pure query selection where both positional and content information come from the encoder.
- Mixed query selection where the content is static and the positional information is dynamic.
Note that for static queries the embeddings can either be learned during training or set to a fixed value.

- [[DETR]]: positional queries are learned during training and content queries are initialized to 0 when passed to the first layer of the decoder.
- [[DN-DETR]], [[DAB-DETR]]: anchors are learned during training and content queries are initialized to 0 when passed to the first layer of the decoder.
- [[Deformable DETR]]: positional and content information is learned during training (this is also static query initialization).
- [[Deformable DETR#Two-Stage Deformable DETR|Two-Stage Deformable DETR]]: the top K encoder features are used from the last layer of the encoder are used to enhance decoder queries. Both the anchors and content are based on a linear transform of the encoder output and are therefore dynamic.
- [[DINO]]: the anchor boxes are initialized using the top K encoder features (similar to Two-Stage Deformable DETR), but the content information is left static (these are learned). This is mixed query initialization.

DINO doesn't use dynamic content information because they claim the dynamic content information may end up containing information about multiple or only part of an object. Instead, they leave the content queries to be learned so that the model is able to use the dynamic positional information to pool more comprehensive content features from the encoder output.

### Look forward twice parameter updates
DINO proposes a new look forward twice scheme to correct the updated parameters with gradients from later layers to leverage the refined box information from later layers to help optimize the parameters of their adjacent early layers.

[[Deformable DETR]] uses an [[Deformable DETR#Iterative Bounding Box Refinement|iterative box refinement ]]where bounding box offsets are predicted at each layer of the decoder relative to the predicted bounding box at the previous stage. Deformable DETR blocks gradient propagation to stabilize training (slight weight tweaks will result in big differences in bounding box predictions from the start to end of the decoder which will cause the weight updates to change too much).

DINO modifies this approach because they think the refined bounding box prediction from the next layer could be useful for updating the adjacent early layer.

They therefore use look forward twice to perform box updates where the parameters of layer-$i$ are influenced by losses of both layer-$i$ and layer-($i+i$).

![[look-forward-twice-parameter-updates.png]]
In look forward once, you predict $b_i^{(\text {pred) }}$ based on a detatched box $b_{i-1}$ and the predicted offsets $\Delta b_i$ so the model weight updates only depends on $\Delta b_i$.

In look forward twice, you use the non-detatched previous box prediction $b_{i-1}'$  and the predicted offsets $\Delta b_i$ so the model weight update depend on the previous prediction and the current prediction.

# Results
![[dino-ablation-study.png]]
Using all three improvements leads to the best results.
- QS means "query selection"
- CDN means "contrastive de-noising"
- LFT means "look forward twice"

As a result, DINO outperforms all previous ResNet-50- based models on COCO val2017 in both the 12-epoch and the 36-epoch set- tings using multi-scale features. Motivated by the improvement, we further ex- plored to train DINO with a stronger backbone on a larger dataset and achieved a new state of the art, 63.3 AP on COCO 2017 test-dev.