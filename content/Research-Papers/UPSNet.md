---
tags: [flashcards]
aliases: [A unified panoptic segmentation network]
source:
summary: bottom-up panoptic segmentation model that resolves the conflicts in 'thing'-'stuff' fusion by predicting an extra unknown class.
---

They add a semantic segmentation and a [[Mask R-CNN]] style [[Instance Segmentation]] head on top of a residual network. These heads operate in parallel and they then add a parameter-free panoptic head that handles merging the logits from each head.

This model has 8 loss functions (vs. [[Panoptic-DeepLab]]'s 3) and has a worse trade off between accuracy and speed.

# Introduction
- Typically fully convolutional neural networks are used for [[Semantic Segmentation]] and proposal based detectors are used for [[Instance Segmentation]].

# Architecture
![[upsnet-architecture.png]]


- They use a single backbone network to provide shared features.
- Two heads work in parallel on these shared features: a semantic head and an instance head.
- An additional panoptic head merges the semantic and instance classifications and allows the model to be trained end-to-end.

### Semantic Head
- Consists of a [[Deformable Convolution]] based sub-network that takes the multi-scale feature from the [[Feature Pyramid Network]] as input.
- The goal of the semantic head is to segment all classes without discriminating instances.
- They use an RoI loss to penalize errors on pixels within instances more. This empircally improved performance of panoptic segmentation without harming the semantic segmentation.

Using [[Deformable Convolution]] makes it so the network can learn where the filter weights should be applied vs. just learning the weights. This could be very useful (in theory) for figuring out where the borders of instances are.

### Instance Head
- Similar to [[Mask R-CNN]]'s design. Outputs class-agnostic masks, and bounding boxes with class labels.

### Panoptic Head
![[upsnet-panoptic-head.mp4]]
[Source](https://youtu.be/LMZI8DDyltQ?t=2804)

![[upsnet-annotated-panoptic-head|700]]

Terms:
- $X$ is the logits from the semantic head with shape ($N_{\text {stuff }}+N_{\text{thing}}$, H, W)
- $N_{\text {stuff}}$, $N_{\text{thing}}$  is the **number** of channels corresponding to "stuff" and "thing" classes respectively.
- $X_{\text{stuff}}$, $X_{\text{thing}}$ are the **channels** of $X$ corresponding to "stuff" and "thing" classes respectively. They have dimension  ($N_{\text {stuff }}$, H, W) and ($N_{\text{thing}}$, H, W).
- $N_{\text{inst}}$ are the number of instances. This determined via mask pruning for inference and uses the number of ground truth instances during training. $N_{\text{inst}}$ can vary depending on the number of instances in an image (unlike $N_{\text{thing}}$ which is constant).
- $Y$ is the mask logits from the instance segmentation head of shape ($N_{\text{inst}}, 28, 28$).
- For any instance $i$, you have mask logits $Y_i$ and bounding box $B_i$ and class ID $C_i$.
- $X_{\text {mask }_i}$ is obtained by taking the values inside box $B_i$ from the channel of $X_{\text {thing }}$ corresponding to $C_i$. It has shape (1, H, W) and all values outside $B_i$ are 0.
- $Y_{\operatorname{mask}_i}$ is obtained by interpolating $Y_i$ to the same scale as $X_{\text {mask }_i}$ and padding zero to achieve a compatible shape with $X_{\text {mask }_i}$.
- $Z_{N_{\text {suff }}+i}$ is the final representation of the $i$th instance and is calculated as $X_{\operatorname{mask}_i}+Y_{\operatorname{mask}_i}$.

The goal of the panoptic head is to produce a tensor $Z$ of shape $\left(N_{\text {stuff }}+N_{\text {inst }}\right) \times H \times W$ and uniquely determine the class and instance ID (if "thing") for each pixel.

Once $Z$ is filled in with all representations of all instances ($\forall i \in N_{\text{inst}}$), you take a softmax along the channel dimension to predict the pixel-wise class. If the maximum value falls into the first $N_{\text{stuff}}$ channel, then it belongs to one of stuff classes. Otherwise the index of the maximum value tells us the instance ID.

**Unknown Class**
Why does UPSNet use an unknown class?
??
[[Panoptic Quality Metric]] is decreased if either FN or FP increases. Therefore, if a wrong prediction is inevitable, predicting such pixel as unknown is preferred since it will increase FN of one class by 1 without affecting FP of the other class.
<!--SR:!2024-06-01,321,250-->

The logits of the extra unknown class are computed as $Z_{\text {unknown }}=\max \left(X_{\text {thing }}\right)-\max \left(X_{\text {mask }}\right)$ where $X_{\text {mask }}$ is the concatenation of all $X_{\text {mask }_i}$ along the channel dimension and has shape $N_{\text {inst }} \times H \times W$. The rationale behind the $Z_{\text {unknown }}$ calculation is that if the maximum of $X_{\text {thing }}$ is larger than the maximum of $X_{\text {mask }}$ then it is likely some instances are missing (FN) since you are more confident that a pixel belongs to a class than that a mask contains it.



- Combines the semantic and instance segmentation results (the per pixel logits).
- Handles the challenges caused by the varying number of instances.
- You can back-prop to the bottom modules (the semantic/instance segmentation heads and the residual network backbone).
- It predicts the final panoptic segmentation via per-pixel classification. It also adds an additional channel corresponding to an extra **unknown class** to help resolve conflicts between semantic and instance segmentation.

# RoI Loss
You take the ground truth bounding boxes of the instances to crop the logits map after the $1 \times 1$ convolution and resize it to $28 \times 28$ (as is done in [[Mask R-CNN]]). The RoI loss is then the cross entropy computed over the $28 \times 28$ patch. This results in more penalty for pixels within instances for incorrect classification (since you don't explicitly penalize incorrect classifications outside of the ground truth RoI).

# Results
Analysis from [[Panoptic SegFormer]]:
> UPSNet (and other methods) "approximate the target task by solving the surrogate sub-tasks, therefore introducing undesired model complexities and suboptimal performance."