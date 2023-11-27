---
tags: [flashcards]
aliases: [Sequential Transformer for Video Instance Segmentation]
source: https://arxiv.org/abs/2112.08275
summary: given an arbitrary long video as input, SeqFormer predicts the classification results, box sequences, and mask sequences in one step without the need for additional tracking branches or hand-craft post-processing.
---

[Code](https://github.com/wjf5203/SeqFormer)

# Abstract
- Uses a stand-alone instance query for capturing a time sequence of instances in a video (unlike [[End-to-End Video Instance Segmentation with Transformers|VisTR]] that uses a query for each instance per frame).
- Attention mechanisms are performed on each frame independently.
- Predicts masks used dynamic convolution where the weights for the convolution are based on the video level instance query.
- Instance tracking comes easily because you have one instance query per agent. You don't need any post-processing or additional branches.

SeqFormer locates an instance in each frame (bounding box query) and aggregates temporal information (weighted sum) to learn a powerful representation of a video-level instance, which is used to predict the mask sequences on each frame dynamically.

# Related Work
There are three main categories of [[Video Instance Segmentation]]:
- Tracking-by-detection: you first make predictions on individual frames and then later associate the predictions temporally. This is sensitive to occlusions and motion blur in videos.
- Clip-level instance masks: you divide a video into multiple overlapping clips and generate mask sequences with clip-by-clip matching on overlapping frames.
- Vision transformers that model instance relationships among video frames (like [[End-to-End Video Instance Segmentation with Transformers|VisTR]]).

### [[End-to-End Video Instance Segmentation with Transformers|VisTR]] vs. SeqFormer
VisTR is the first method that adapts Transformer to the VIS task. However, VisTR has a fixed number of input queries hardcoded by video length and maximum number of instances. Each query corresponds to an object on every single frame. In SeqFormer, instances with the same identity ==share the same query==.
<!--SR:!2023-12-13,230,310-->

[[IFC]] improves on VisTR and builds communication between frames in the transformer encoder by not flattening the space-time feature (T, H, W) into one dimension, but it still flattens them in the decoder. SeqFormer operates independently on each frame which lets the model attend to locations following the movement of the instance.

# Approach
### Single instance query across time + per-frame attention
Unlike [[End-to-End Video Instance Segmentation with Transformers|VisTR]] use a single instance query per-agent, but attention process is done independently one each frame. This aligns with the conclusion from action recognition work where "the 1D time domain and 2D space domain have different characteristics and should be handled in a different fashion."

They found using a single instance query was sufficient although an object may be of different positions, sizes, shapes, and various appearances.

Since the instance changes appearance and position, the model should attend to different exact spatial locations of each frame and attention is performed on each frame independently.

> [!NOTE] Human motivation
> Given a video, humans can effortlessly identify every instance and associate them through the video, despite the various appearance and changing positions on different frames. If an instance is hard to recognize due to occlusion or motion blur in some frames, humans can still re-identify it through the context information from other frames. In other words, for the same instance on different frames, humans treat them as a whole instead of individuals.

### Box Queries
![[seqformer-box-queries.png]]

Each transformer decoder layer predicts a box query per-frame that is a refinement from the previous layer's predicted box query (or the initial instance query for the first layer). Each frame's query is kept independent (you only refine based on the respective frame from the previous layer's box queries). By repeating this refinement through decoder layers, SeqFormer locates the instance in each frame in a coarse-to-fine manner, in a similar way to Deformable DETR.

At the last layer of the decoder, the box queries are used to predict the bounding boxes.

### Dynamic Mask Head
SeqFormer uses a dynamic mask head where you predict the weights to apply to the input at inference time. This lets you modify the weights of the convolutions depending on the input image.

This was introduced in CondInst and used in QueryInst as well.

Dynamic mask heads share the same mask head parameters for an instance across frames, but have different parameters for each instance.

# Architecture
![[seqformer-architecture.png]]
- You first pass the input frames to a backbone that extracts multi-scale feature maps from each image independently.
- You then pass these feature maps to a Deformable DETR transformer encoder to enrich the feature maps. This operates on each frame independently by keeping the spatial and temporal dimensions of feature maps seperate rather than flattening them into one dimension as [[End-to-End Video Instance Segmentation with Transformers|VisTR]] does (which means the positional encodings need to be 2D vs. 3D).
- The enriched feature maps are passed to the transformer decoder which takes the instance query (learned embeddings) as input and produces a video-level instance representation and a set of $T$ bounding box queries.
- Three output heads are used for instance classification (based on video-level instance representation), instance sequence segmentation (based on the video-level instance representation and transformer encoders), and bounding box prediction (based on the box queries).
- The parameters for the mask head's kernels are predicted based on the video-level instance representation and the kernel is applied to transformer encoder features that pass through a mask branch to combine the multi-scale feature maps.

### Transformer Encoder
- Apply a 1x1 convolution to reduce the channel dimension of all feature maps to $C = 256$. These features are denoted $\left\{\mathbf{f}_t^{\prime}\right\}_{t=1}^T, \mathbf{f}_t^{\prime} \in \mathbb{R}^{C \times H^{\prime} \times W^{\prime}}, t \in[1, T]$
- You then add fixed positional encodings and the encoder performs deformable attention on the feature map. The spatial and temporal dimensions of feature maps are retained rather than flattening them into one dimension. The feature maps with positional encodings are denoted $\left\{\mathbf{f}_t\right\}_{t=1}^T$.

### Query Decompose Transformer Decoder
- They use a fixed number of Instance Queries to query the features of the same instance from each frame.
- They decompose the instance query into $T$ frame-specific box queries which each serve as an anchor for retrieving and location features on the corresponding frame.

At the first decoder layer, an instance query $\mathbf{I}_q \in \mathbb{R}^C$ is used to query the instance features on feature maps of each frame independently using deformable attention: 
$$\mathbf{B}_t^1=\operatorname{DeformAttn}\left(\mathbf{I}_q, \mathbf{f}_t\right)$$
where $\mathbf{B}_t^1 \in \mathbb{R}^C$ is the box query on frame $t$ from the 1-st decoder layer.

For all decoder layers after the first one $(l > 1)$, the box query from the last layer for this frame is used as input:
$$\mathbf{B}_t^l=\operatorname{DeformAttn}\left(\mathbf{B}_t^{l-1}, \mathbf{f}_t\right)$$

> [!NOTE] 
> For the $l > 1$ layer, the box queries are refined based on the previous layer's box query output for the respective frame. There is no temporal mixing of the predicted box information.

After **each** decoder layer, the instance instance query is re-calculated using a weighted sum of the predicted boxes on all frame for this instance + the updated instance query from the previous frame ($\mathbf{I}_q^{l-1}$):
$$\mathbf{I}_q^l=\frac{\sum_{t=1}^T \mathbf{B}_t^l \times \mathrm{FC}\left(\mathbf{B}_t^l\right)}{\sum_{t=1}^T \mathrm{FC}\left(\mathbf{B}_t^l\right)}+\mathbf{I}_q^{l-1}$$
The weights for the above calculation are learned based on the box embedding (denoted $\text{FC}$).

After all $N_d$ decoder layers, you get a final instance query and $T$ box queries (on for each frame). The instance query is a **video-level instance representation** since it contains information from all frames and is denoted $I_q^{N_d}$ and called the **output instance embedding**. The **box queries** are denoted $\left\{\mathbf{B E}_t\right\}_{t=1}^T$ where $\mathbf{B E}_t \in \mathbb{R}^{N \times d}$ and you have one for each frame ($T$ total).

### Class Head
A linear projection is used as the class head. It takes the instance embedding from the transformer decoder and predicts a class probability.

### Box Head
A 3-layer [[Linear|FFN]] is used for the box head. For the bounding box prediction $\mathbf{B E}_t$ of each frame, the FFN predicts the normalized center coordinates, height, and width of the box w.r.t the frame.

### Mask Prediction
**Mask Branch**:
![[mask-branch-and-mask-head.png]]
There is a **mask branch** that is an [[Feature Pyramid Network|FPN]] like architecture that takes the multi-scale feature maps from the transformer as input and then produces a feature map sequence $\left\{\hat{\mathbf{F}}_{\text {mask }}^1, \hat{\mathbf{F}}_{\text {mask }}^2, \ldots, \hat{\mathbf{F}}_{\text {mask }}^T\right\}$ that is 1/8 of the input resolution and has 8 channels for each frame independently.

**Relative coordinates**:
You then concatenate a map of the relative coordinates that contain the distance from the center of the predicted bounding box $\hat{\mathbf{b}}_{(\sigma(i), t)}$ for the corresponding frame (you generate an evenly spaced grid of points and then subtract the coordinates of the center box). This provides spatial information about the distance from the center of the object for the $1 \times 1$ convolutions which otherwise don't have spatial context. These feature maps are denoted $\{\mathbf{F}^t_{\text{mask}}\}^T_{t=1}$ where $\mathbf{F}^t_{\text{mask}} \in \mathbb{R}^{10 \times \frac{H}{8} \times \frac{W}{8}}$.

An example of what the computed grid looks like is:
![[seqformer-20230323125035907.png]]

If you have a BBOX centered at (100, 200) and then do:
```python
locations = torch.tensor([100, 200]) - locations
```

you get the points in red shown below: 
![[bbox_minus_locations.png]]

[Generated with this notebook](https://colab.research.google.com/drive/1-FQeQzSf2-2FnyXWCAslOz9TqxLuIcIO?usp=sharing)

**Mask Head**:
They use dynamic convolution for the mask head.

They use the video-level instance embedding as input and use a 3-layer [[Linear|FFN]] to encode the embedding $I_{\sigma(i)} \in \mathbb{R}^C$ into parameters $w_i$ of mask head corresponding to the instance with index $\sigma(i)$. 

The mask head has three $1 \times 1$ convolution layers and the same instance will use the same convolution weights on all frames. It uses the predicted weights $w_i$ and the corresponding feature map with relative coordinates $\mathbf{F}^t_{\text{mask}}$ to perform $1 \times 1$ convolutions and predict the mask sequences:
$$\left\{\mathbf{m}_i^t\right\}_{t=1}^T=\left\{\operatorname{MaskHead}\left(\mathbf{F}_{\text {mask }}^t, \omega_i\right)\right\}_{t=1}^T$$

# Matching and Loss
The decoder predicts a fixed size of $N$ predictions where $N$ is significantly larger than the number of items.

They compute a bipartite graph matching between the predictions and ground truth using the class label and bounding box predictions (they don't include the mask in the matching cost because this is too computationally expensive). 

They define:
- $\mathbf{y}$ is the ground truth set of video-level instances
- $\hat{\mathbf{y}}_i=\left\{\hat{\mathbf{y}}_i\right\}_{i=1}^N$ is the predicted instance set.
- Each element $i$ of the ground truth set is represented as $\mathbf{y}_i=\left\{\mathbf{c}_i,\left(\mathbf{b}_{i, 1}, \mathbf{b}_{i, 2}, \ldots, \mathbf{b}_{i, T}\right)\right\}$.
- $\mathbf{c}_i$ is the target class label (including $\varnothing$) 
- $\mathbf{b}_{i, t} \in[0,1]^4$ is a vector that defines the ground truth bounding box.
- $\hat{p}_{\sigma(i)}\left(\mathbf{c}_i\right)$ is the predicted class label.
- $\hat{\mathbf{b}}_{\sigma(i)}$ is the predicted bounding box.

The pair-wise matching cost between ground truth $y_i$ and a prediction with $\sigma(i)$ is given by:
$$\mathcal{L}_{\text {match }}\left(\mathbf{y}_i, \hat{\mathbf{y}}_{\sigma(i)}\right)=-\hat{p}_{\sigma(i)}\left(\mathbf{c}_i\right)+\mathcal{L}_{\text {box }}\left(\mathbf{b}_i, \hat{\mathbf{b}}_{\sigma(i)}\right)$$
where $\mathbf{c}_i \neq \varnothing$.

Given the optimal assignment from the Hungarian Algorithm, they then compute the loss for all matched pairs to train the network.
$$\begin{aligned}
\mathcal{L}_{\text {Hung }}(\mathbf{y}, \hat{\mathbf{y}})=\sum_{i=1}^N & {\left[-\log \hat{p}_{\hat{\sigma}(i)}\left(\mathbf{c}_i\right)+\mathbb{1}_{\left\{\mathbf{c}_i \neq \varnothing\right\}} \mathcal{L}_{\text {box }}\left(\mathbf{b}_i, \hat{\mathbf{b}}_{\hat{\sigma}}(i)\right)\right.} \\
& \left.+\mathbb{1}_{\left\{\mathbf{c}_i \neq \varnothing\right\}} \mathcal{L}_{\text {mask }}\left(\mathbf{m}_i, \hat{\mathbf{m}}_{\hat{\sigma}}(i)\right)\right] .
\end{aligned}$$


> [!WARNING] The paper uses focal loss instead of negative log likelood loss for calculating the class loss.
> The paper claims to use $-\hat{p}_{\sigma(i)}\left(\mathbf{c}_i\right)$ for the class loss (as is written in formulas above). This is negative log likelihood loss (NLLLoss). `nn.Sequential([LogSoftmax(), NLLLoss()])` and is equivalent to `CrossEntropyLoss` if applied to a model's logits. In [[DETR]] Cross Entropy Loss is used in the implementation. [[Deformable DETR]] changes this to focal loss (mentioned in experiment settings). SeqFormer uses the changes from [[Deformable DETR]] so they also use focal loss.

### Bounding Box Loss
Is a combination of [[Regularization|L1 Loss]] and generalized [[IoU]] loss.

### Mask Loss
The masks from the mask head are 1/8 of the video resolution which may lose some details. Therefore, they upsample the predicted masks to 1/4 the original resolution and downsample the ground truth mask to the same resolution.

The mask loss is a combination of Dice and Focal Loss.

### VIS Metrics
Video instance segmentation is evaluated by the metrics of average precision (AP) and average recall (AR).

To evaluate the spatio-temporal consistency of the predicted mask sequences, the IoU computation is carried out in the spatial-temporal domain. This requires a model not only to obtain accurate segmentation and classification results at frame-level but also to track instance masks between frames accurately.

# Implementation Details
### Model Settings
- ResNet-50 is used as the backbone. The last three stages {C3, C4, C5} are used with strides {8, 12, 32} respectively.
- They generate an additional feature map C6 via a $3 \times 3$ stride 2 convolution on C5.
- For deformable attention they use the number of keypoits as $K = 4$ and 8 attention heads.
- They use six encoder and six decoder layers with hidden dimension 256 for the transformer. They set the number of instance queries to 300.

### Training
- AdamW with base learning rate of $2 \times 10^{-4}$ and learning rates for the backbone and linear projections are scaled by a factor of 0.1.
- Pre-train on COCO by setting the number of input frames to 1.
- The model is trained on 8 V100 GPUs with 32 GB Ram with 2 5-frame video clips per GPU.

# Results
The below figure shows SeqFormer significantly outperforms the previous method with similar parameters.
![[seqformer-performance-vs-model-size.png]]
The best performing model uses a [[Swin Transformer|Swin]] backbone, but also has the most parameters.

When using a ResNet-50 backbone (as well as other backbones), SeqFormer outperforms [[End-to-End Video Instance Segmentation with Transformers|VisTR]] and [[MaskTrack R-CNN]] with fewer params and better FPS. SeqFormer also outperforms and is faster than [[CrossVIS]] but it uses more params.
![[seqformer-results-resnet50.png]]

# Ablation Studies
- Combining spatial and temporal dimensions hurts performance.

**BBOX Query Combination**
- Using a learned weighted combination of BBOX queries to generate the video level instance query performed better than directly summing or averaging (sum and then divide by number of frames). 
- Directly summing will cause the value to be unstable with different frame numbers.
- Averaging will cause information to be diluted if an instance only appears in some of the frames.

# Updated Architecture
![[safetynet-seqformer-modifications.excalidraw|800]]