---
tags: [flashcards]
source: https://arxiv.org/abs/1911.10194
summary: a model for panoptic segmentation.
---

This model adds an additional decoder to [[DeepLabv3+]] (which does semantic segmentation) in order to do [[Panoptic Segmentation]].

# Related Work
There are two main categories of work: top-down and bottom-up. The former first generates proposals for masks and then the masks are refined. The latter, directly predicts the masks without a secondary stage.

### Top-Down
Top-Down approaches (like [[UPSNet]]) add an additional [[Semantic Segmentation]] branch to [[Instance Segmentation]] models like [[Mask R-CNN]]. These approaches are sometimes referred to as two-stage methods since they require an additional stage to generate proposals (they firwt g.

They have an **issue with generating overlapping instance masks as well as duplicate pixel-wise semantic predictions**. These issues need to be resolved using heuristics or advanced modules both of which are often slow in speed resulting from multiple sequential processes in the pipeline.

![[panoptic-deeplab-fig8-masks-predicted-for-human.png]]
For instance, in the above image, if you predicted instance masks first you would have the mask for "beer" overlapping the mask for "person." Additionally, the pixels corresponding to the "beer" would have both semantic (class) labels. This conflict needs to be resolved in the post-processing step.

Conflicts between overlapping masks are often handled by their predicted confidence or relationships between categories (ex. "ties" should always be in front of "people"). If the overlap occurs between "things" and "stuff" predictions (ex. the bottom of the "person" overlaps with the "floor" mask), the "things" mask is favored. 

**Ex: using Mask R-CNN for panoptic segmentation**:
You first predict overlapping instance segmentations. You then do some post-processing to resolve mask overlaps.

> [!NOTE] The instance segmentation task allows overlapping masks.
> This is an issue for modifying instance segmentation models for panoptic segmentation since you have to handle the overlap.
> 
> The overlap is allowed for instance segmentation since it's possible one instance is behind another. It also allows predicting multiple instances in parallel since overlap is allowed. However, by the definition of panoptic segmentation, you are only allowed to have one label per-pixel.

You can then fill the remaining regions without masks with a light-weight stuff segmentation branch.

This is how you address "stuff" not having a concept of an individual entity (i.e., you can't have individual "sky" or "grass").

### Bottom-Up
Bottom-up approaches typically start with a semantic segmentation predictions followed by grouping operations to generate instance masks.

This approach allows for a simple and fast scheme (such as majority vote) to merge semantic and instance segmentation results. Additionally, you don't have the issue of needing to handle overlapping segments as you do in the top-down approaches since each pixel only can belong to a single mask/class.

Bottom-up approaches could be faster than top-down but until this paper there weren't any strong contenders.

# Architecture
![[Annotated Panoptic-DeepLab Architecture|900]]
- [[DeepLabv3#Spatial Pyramid Pooling]] is used to get multi-scale context (the improved ASPP from DeepLabv3 with image pooling is used).
- Instance segmentation is obtained by predicting the object centers and regressing every pixel with a "thing" class label to its corresponding center ("stuff" pixels don't have a concept of a center).
- The semantic segmentation and class-agnostic instance segmentation are then fused to generate panoptic segmentation results.

### Encoder Backbone
![[panoptic-deeplab-encoder-backbone.png|200]]
An encoder backbone is shared for extracting features for both the semantic and instance branches.

[[Dilated Convolution]] is used in the last block of the network backbone to extract a denser feature map (keeping the resolution at 1/16 vs. 1/32 of the input dimensions).

### Dual-ASPP
![[panoptic-deeplab-dual-aspp.png|200]]
The ASPP modules are typical of semantic segmentation models (ex. [[DeepLab]]) since they allow you to extract features with multiple FOV.

Seperate ASPP and decoder modules are used for the semantic/instance segmentation branches with the hypothesis that the two branches require different contextual and decoding information (and this was verified empirically).

### Dual-Decoder
![[panoptic-deeplab-dual-decoder.png|300]]
A lightweight decoder is used for each branch and consists of a single convolution during each upsampling stage. The decoder is similar to [[DeepLabv3+#Decoder|DeepLabv3+'s Decoder]] with two modifications:
- An additional low-level feature with output stride 8 is added to the decoder (in addition to the low-level feature with output stride 4).
- In each upsampling stage they use a single $5 \times 5$ [[Depthwise Separable Kernels]].

> [!question] Where does the low-level feature come from?

### Semantic Segmentation Head
![[semantic-segmentation-head.png|200]]
- You have a $\text{NUM\_CLASSES} \times H \times W$ output where each pixel has a distribution over the classes predicted. It predicts for both "thing" and "stuff" classes.
- They use a [[DeeperLab#Weighted Bootstrapped Cross-Entropy Loss|Weighted Bootstrapped Cross-Entropy Loss]] from [[DeeperLab]] that focuses on hard pixels and pixels that belong to small instances.

### Instance Segmentation Head
![[instance-segmentation-head.png|200]]
The instance segmentation branch involves a simple instance center regression where the model learns to predict instance centers as well as the offset from each pixel to its corresponding center (the offset is predicted seperately). This results in a simple grouping operation (no advanced clustering needed) to assign pixels to their closest predicted center.

The instance segmentation is done class-agnostic (you don't have a concept of the class).

During training, groundtruth instance centers are encoded by a 2D Gaussian with a standard deviation of 9 pixels. This makes it so you don't have to perfectly match the objects center. They use [[Mean Squared Error|MSE]] loss to minimize the distances between predicted heatmaps and the 2D Gaussian-encoded groundtruth heatmaps.

They use [[Regularization|L1 Loss]] for the offset prediction, which is only activated at pixels belonging to object instances.

During inference, predicted "thing" pixels are grouped to their closest predicted mass center foring class-agnostic instance segmentation.

# Panoptic Segmentation
![[fusing-instance-and-semantic-predictions.png]]
Instances are represented by their center of mass $\left\{\mathcal{C}_n:\left(i_n, j_n\right)\right\}$. To get the center point predictions, they first use keypoint-based [[Non-maximum Suppression]] to reduce the number of centerpoints. This is equivalent to applying max pooling on the heatmap predictions and keeping only values that don't change before and after pooling (these are the maximum values). In the experiment, they used max-pooling with kernel size 7.

After filtering with max pooling, a hard threshold is used to filter out predictions with low confidence (they used $0.1$) and then only locations with the top-$k$ highest confidence scores are kept (they used $k = 200$).

For each "thing" pixel the offset to its center is predicted (horizontal, vertical offset) so the instance id for the pixel is just the closest center after moving the pixel location $(i, j)$ by the predicted offset $\mathcal{O}(i, j)$. **No center is predicted for "stuff" pixels whose instance ids are always set to 0.**

The class label for an instance mask is based on the majority vote of the corresponding predicted semantic labels. This can be efficiently implemented on the GPU since you can accumulate the class label histograms in parallel.

For the instance mask predictions, you can compute a confidence score via:
$$\text {Score(Objectness)} \times \text {Score}(\text{Class})$$
where $\text { Score(Objectness) }$ is the unnormalized objectness score from the class-agnostic center point heatmap and $\text { Score(Objectness) }$ is the average of the semantic segmentation predictions within the predicted mask region for that class.

# Loss Overview
Panoptic-DeepLab is trained with three loss functions:
- Weighted bootstrapped cross entropy loss for the semantic segmentation head ($\mathcal{L}_{h e a t m a p}$).
- MSE loss for the center heatmap head ($\mathcal{L}_{heatmap}$).
- L1 loss for the center offset head ($\mathcal{L}_{\text{offset}}$).
And the total loss is calculated as:
$$\mathcal{L}=\lambda_{\text {sem }} \mathcal{L}_{\text {sem }}+\lambda_{\text {heatmap }} \mathcal{L}_{\text {heatmap }}+\lambda_{\text {offset }} \mathcal{L}_{\text {offset }}$$
where:
- $\lambda_{s e m}=3$ for pixels belonging to instances with small areas (smaller than $64 \times 64$ as in [[DeeperLab]]) and $\lambda_{s e m}=1$ elsewhere.
- To make sure the losses had similar magnitudes, they set $\lambda_{\text {heatmap }}=200$ and $\lambda_{o f f s e t}=0.01$.

# Results
### PQ vs. Inference Speed
Panoptic-DeepLab gets the best speed-accuracy trade off across COCO, Cityscapes, and Mapillary compared to [[UPSNet]] and [[DeeperLab]].

![[pq-vs-inference-time-cityscapes.png]]

![[pq-vs-inference-time-cityscapes.png.png]]

Panoptic-DeepLab has a better accuracy to speed trade-off than UPSNet, but it does get a lower PQ on COCO (note different input sizes are used). It performs better on Cityscapes where input sizes are the same.

![[panoptic-deeplab-20221231151132729.png]]
![[panoptic-deeplab-20221231151138670.png]]

### $\mathbf{P Q}^{\text {Thing }}$ vs. $\mathbf{P Q}^{\text {Stuff }}$
Panoptic-DeepLab has a higher PQ for stuff than things compared to top-down approaches that can better handle scale variation. [[MaX-DeepLab]] is likely able to improve on this since attention is a global operation.

### Future Work
Current bottom-up pantoptic segmentation still requires post-processing steps (ex. filtering the center heatmap and associating pixels to their instances). This make make it hard to end-to-end train the whole system.