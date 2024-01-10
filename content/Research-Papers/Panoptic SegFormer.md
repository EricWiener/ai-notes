---
tags:
  - flashcards
aliases:
  - Delving Deeper into Panoptic Segmentation with Transformers
source: https://arxiv.org/abs/2109.03814
summary: a general framework for panoptic segmentation with transformers.
publish: true
---

> [!NOTE] Main concepts introduced in framework:
> 1. A deeply-supervised mask decoder.
> 2. A query decoupling strategy.
> 3. Improved post-processing method.

- Uses [[Deformable DETR]] to process multi-scale features.

# Framework
### Deeply-supervised mask decoder
The mask decoder is responsible for predicting binary masks and class label. During training, predictions are made using the output of each layer of the mask decoder (vs. just using the final layer's output). The loss is calculated for all layers and this is what the paper refers to as "deep supervision".

Some literature argue that the reason [[DETR|DETR]] is slow to converge is because the attention maps pay equal attention to all pixels in the feature map (vs. focusing on important regions). Panoptic SegFormer's deep supervision strategy lets the attention modules quickly focus on meaningful semantic regions and capture meaningful information in earlier stages. It improves performance and reduces the number of required training epochs by half compared to [[Deformable DETR]].

### Query decoupling strategy
- Panoptic SegFormer splits it's $N$ queries into $N_{\text{things}}$ + $N_{\text{stuff}}$ queries.
- $N_{\text{things}}$ is a hyperparameter that dictates the total number of instances that can be detected in an image (irrespective of the number of classes).
- $N_{\text{stuff}}$ is the number of "stuff" classes.
- The thing queries first pass through a location decoder while the stuff queries go directly to the mask decoder. Treating the queries differently was motivated by the belief "things" and "stuff" should be treated differently.
- The thing and stuff queries can use the same pipeline once they are passed to the mask decoder which accepts both thing queries and stuff queries and generates the final masks and categories.
- During training, ground truths are assigned via bipartite matching to thing queries. For stuff queries, a class-fixed assign strategy is used with each stuff query corresponding to one stuff category.

### Post-processing method
- Post-processing strategy improves performance without additional costs by jointly considering classification and segmentation qualities to resolve conflicting mask overlaps.
- This was motivated because common post-processing (ex. pixel-wise argmax to decide on the class) tends to generate false-positive results due to extreme anomalies. 

# Architecture
![[panoptic-segformer-20230104122130841.png]]
Panoptic SegFormer consists of three key modules: transformer encoder, location decoder, and mask decoder.

### Backbone
![[panoptic-segformer-annotated-backbone|900]]
- You pass an input image $X \in \mathbb{R}^{H \times W \times 3}$ to the backbone network and get feature maps $C_3, C_4, C_5$ from the last three stages. These have resolutions $1/8, 1/16, 1/32$ with respect to the original image.
- You then project each of the feature maps from however many channels they have (denoted with `?`) to 256 channels using a 1x1 conv (the paper calls this a fully-connected layer).
- You then flatten each of the feature maps and concatenate them to produce your input sequence to the encoder. 

### Transformer Encoder
The transformer encoder is used to refine the multi-scale feature maps sequence given by the backbone.

Because of the high computational cost of self-attention, previous transformer-based methods only processed low-resolution feature maps (ex. ResNet $C_5$) in their encoders which limited segmentation performance (higher resolution features are needed to get better results). 

Panoptic SegFormer uses [[Deformable Attention]] in the transformer encoder which has a lower computational complexity (it focuses on a fixed number of locations with learned offsets vs. all locations). Therefore, they are able to use high-resolution and multi-scale feature maps.

They form tokens for each pixel in the multi-scale feature maps where each token has dimension 256. They define $L_i$ to be $\frac{H}{2^{i+2}} \times \frac{W}{2^{i+2}}$ and the shapes of the flattened feature maps are $L_1 \times 256, L_2 \times 256, \text { and } L_3 \times 256$. They then pass $L_1 + L_2 + L_3$ tokens to the transformer encoder where each token has length 256.

### Location Decoder
![[annotated-location-decoder.png|400]]

Location information is important to distinguish between difference instances. In order to include location information, the location decoder adds location information of "things" (not "stuff") into the learnable queries used by the mask decoder.

Given $N_{\text{thing}}$ randomly initialized queries and the outputs of the encoder, the location decoder will output $N_{\text{thing}}$ location-aware queries using cross-attention.

During training an auxiliary MLP head is added on top of the location-aware queries to predict the bounding boxes and categories of the target object. **This branch (shown in red in the diagram above) is discarded during inference.**

### Mask Decoder
![[annotated-mask-decoder.png|400]]
The mask decoder is for final classification and segmentation. It takes the $N_{\text{thing}}$ tokens with location information from the location decoder and the $N_{\text{stuff}}$ randomly initialized queries. The $N_{\text{stuff}}$ queries don't pass through the location decoder since there is no concept of an instance for "stuff."

![[screenshot-2023-01-04_14-04-59.png]]
- Queries $Q$: these are the $N_{\text{things}}$ location aware thing queries and $N_{\text{stuff}}$ class-fixed stuff queries.
- Keys $K$, Values $V$: these are projected from the output of the encoder (denoted $F$).
- Prediction is split into category classification done via a FC layer and a binary mask prediction done via a $1 \times 1$ convolution.

**Category classification**:
Category classification is performed through a FC layer on top of the output of each decoder layer (denoted $Q_{\text{refine}}$). Each thing query needs to predict probabilities over all thing categories. Each stuff query only predicts the probability of its corresponding stuff category.

The paper uses a very lightweight FC head (only 200 parameters) to make the class predictions in the hope that ==this ensures the semantic information (aka class information) of the attention maps is highly related to the mask==.
<!--SR:!2024-03-29,292,274-->

**Binary mask prediction**:
> [!TODO] what are the shapes of the attention maps?

"Intuitively, the ground truth mask is exactly the meaningful region on which we expect the attention module to focus." Therefore, the paper uses the ==attention maps== as the input to a $1 \times 1$ conv that predicts the binary mask.
<!--SR:!2025-06-05,612,277-->

Before the attention maps $A$ are passed to the $1 \times 1$ conv, they are first split and reshaped into attention maps $A_3, A_4, A_5$ which have the same spatial dimensions as $C_3, C_4, C_5$ (the backbone network feature maps). This is denoted:
$$(A_3, A_4, A_5) = \text{Split}(A)$$
Then, the attention maps are upsampled to the resolution $(H/8, W/8)$ which corresponds to the highest resolution backbone feature map, $C_3$. The feature are then concatenated along the channel dimension once they all have the same resolution:

$$A_{\text {fused }}=\operatorname{Concat}\left(A_1, \operatorname{Up}_{\times 2}\left(A_2\right), \operatorname{Up}_{\times 4}\left(A_3\right)\right)$$
where $\mathrm{Up}_{\times 2}(\cdot)$ and $\mathrm{Up}_{\times 4}(\cdot)$ mean the 2 times and 4 times bilinear interpolation operations.

Finally, the fused attention maps $A_{\text{fused}}$ are passed to a $1 \times 1$ conv for binary mask prediction.

**Training**:
During training you predict a mask and a category at each layer of the decoder. This is referred to as ==deep supervision== and using it results in the mask decoder performing better and convering faster.
<!--SR:!2024-11-29,336,290-->

**Inference**:
During inference only the mask and category predictions from the last layer of the 
decoder are used. 

# Mask Merging
Most papers will produce an output of $(C, H, W)$ where $C$ is a 0-1 distribution over the classes that can be assigned to. The papers then take a pixel-wise argmax to decide on the label for a particular pixel. The paper observed that **this consistently produces false-positive results due to abnormal pixel values.** Instead, they use a ==mask-wise merging strategy== by resolving conflicts between predicted masks. They give precedence to masks with the highest confidence scores. The pseudo-code is shown below:

![[Excalidraw/panoptic-segformer-mask-merging-algo.excalidraw.png]]

The mask-wise merging strategy takes $c$, $s$, and $m$ as input, denoting the predicted categories, confidence scores, and segmentation masks, respectively. It outputs a semantic mask `SemMsk` and an instance id mask `IdMsk`, to assign a category label and an instance id to each pixel. Specifically:
- `SemMsk` and `IdMsk` are first initialized by zeros.
- Sort prediction results in descending order of confidence score and fill the sorted predicted masks into `SemMsk` and `IdMsk` in order.
- Discard the results with confidence scores below $t_{\text{cnf}}$ and remove the overlaps with lower confidence scores. Only masks with a remaining fraction of the original mask will be kept (above $t_{\text{keep}}$). 
- Finally, the category label and unique id of each mask are added to generate non-overlap panoptic format results.
<!--SR:!2024-09-28,475,317-->

**Confidence Score Calculation**:
The confidence score for a mask takes into account classification probability and predicted mask quality. The confidence score of the $i$-th result is given by:
$$s_i=p_i^\alpha \times \operatorname{average}\left(\mathbb{1}_{\left\{m_i[h, w]>0.5\right\}} m_i[h, w]\right)^\beta$$
- $p_i$ is the most likely class probability of the $i$-th result.
- $m_i[h, w]$ is the mask logit at pixel $[h, w]$.
- $\alpha, \beta$ are hyperparameters used as exponents to balance the weight of classifcation probability and segmentation quality. Since the classification quality and predicted mask quality terms are both $\in [0, 1]$, having a higher exponent will result in that term's weight being decreased.

**Pixel-wise Max vs. Mask-Wise Merging**
![[pixel-wise-argmax-vs-mask-merging-viz.png]]
When using pixel-wise argmax to decide the labels the inference results are very messy. The results are much better when using the mask-wise merging strategy.

# Loss Functions
### Overall loss ($\mathcal{L}$)
The overall loss for Panoptic SegFormer is:
$$\mathcal{L}=\lambda_{\text {things }} \mathcal{L}_{\text {things }}+\lambda_{\text {stuff }} \mathcal{L}_{\text {stuff }}$$
where $\lambda_{\text {things }}, \lambda_{\text {stuff }}$ are hyperparamters and $L_{\text {things }}, L_{\text {stuff }}$ are the losses for things and stuff.

### Detection loss ($\mathcal{L}_{\text{det}}$)
This is the loss of [[Deformable DETR]] that is used to perform detection (bounding box loss). 

![[Deformable DETR#Loss]]

### Things loss ($\mathcal{L}_{\text{things}}$)
This is the overall loss function used for the thing categories:
$$\mathcal{L}_{\text {things }}=\lambda_{\text {det }} \mathcal{L}_{\text {det }}+\sum_i^{D_m}\left(\lambda_{\text {cls }} \mathcal{L}_{\text {cls }}^i+\lambda_{\text {seg }} \mathcal{L}_{\text {seg }}^i\right)$$
- The $\lambda$ values are weights to balance the three losses.
- $D_m$ is the number of layers in the mask decoder (the loss is calculated for all layers because of the deep supervision).
- $\mathcal{L}_{\mathrm{cls}}^i$ is the classification loss implemented by Focal Loss.
- $\mathcal{L}_{\mathrm{seg}}^i$ is the segmentation loss implemented by Dice Loss.
- $\mathcal{L}_{\mathrm{det}}$ is the focal loss for BBOX classification used by [[Deformable DETR]].

### Stuff loss ($\mathcal{L}_{\text{stuff}}$)
This is the loss for stuff. There is a one-to-one mapping between stuff queries and stuff categories and no BBOX loss is needed.
$$\mathcal{L}_{\mathrm{stuff}}=\sum_i^{D_m}\left(\lambda_{\mathrm{cls}} \mathcal{L}_{\mathrm{cls}}^i+\lambda_{\mathrm{seg}} \mathcal{L}_{\mathrm{seg}}^i\right)$$
Like the loss for things,
- $\mathcal{L}_{\mathrm{cls}}^i$ is the classification loss implemented by Focal Loss.
- $\mathcal{L}_{\mathrm{seg}}^i$ is the segmentation loss implemented by Dice Loss.

# Results
- It is able to train much faster than [[DETR|DETR]] (24 epochs vs. 300+)

![[table-10-results-of-each-layer-mask-decoder.png|400]]
Using just the first two layers of the decoder has similar performance to using the entire mask decoder and runs much faster.

# Training
- We employ a threshold 0.5 to obtain binary masks from soft masks.
- During the training phase, the predicted masks that be assigned ∅ will have a weight of zero in computing Lseg.