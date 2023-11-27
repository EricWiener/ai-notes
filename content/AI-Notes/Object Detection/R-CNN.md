# Selective Search: Region Proposal
Selective search generates a small number of region proposals that are likely to contain an object (2000 proposals in a few seconds on CPU). Then we only run our classifier on these likely regions. 

These approaches use hand-crafted features. For instance, you could look for areas that have a lot of edges. Eventually they are replaced by neural networks.

# R-CNN: Region proposals + CNN features

![[r-cnn-overview.mp4]]

> [!note] Overview
> The R-CNN object detector processes region proposals with a CNN
> 
> At test-time, eliminate overlapping detections using non-max suppression (NMS)
> 
> Evaluate object detectors using mean average precision (mAP)
This detector gets region proposals from **selective search**. It then crops these images to squares and inputs them to a CNN. The output of the CNN is then passed through a linear classifier to get the final labels and a bounding box regression (to correct bounding box proposed via region proposal).

![[screenshot-2022-03-20_14-54-00.png]]
![[AI-Notes/Object Detection/r-cnn-srcs/Screen_Shot 2.png]]

You use bounding-box regression to shift the bounding box to a better choice (you can have a model trained on bounding box info to do this). The regression is performed without knowledge of what the classification is (a parallel head to classification).

## Pipeline:
1. Run region proposal method to compute ~2000 proposals (aka region of interest/RoI). Each has center $(p_x, p_y)$ and dimensions $(p_w, p_h)$.
2. Resize each region to 224x224
3. Pass each region through a ConvNet back-bone (ConvNet uses same weights for all proposals).
4. Predict class scores and bounding box transform using parallel heads.
    1. Predict class scores for each region
    2. Perform a bounding box transform (invariant to where the box is in the image). This predicts a transform $(t_x, t_y, t_w, t_h)$ to apply to the corresponding region proposal.
5. Use scores to select a subset of region proposals. You could emit boxes with a low background score, make per-class thresholds, etc.
6. Compare with ground-truth boxes

## R-CNN Bounding Box Regression:
- Consider regional proposal with center $(p_x, p_y)$, width $p_w$, and height $p_h$.
- Model predicts a transform $(t_x, t_y, t_w, t_h)$ to correct the region proposal.

The output box is:
$b_{x}=p_{x}+p_{w} t_{x}$
$b_{y}=p_{y}+p_{h} t_{y}$
$b_{w}=p_{w} \exp \left(t_{w}\right)$
$b_{h}=p_{h} \exp \left(t_{h}\right)$

**This is because:**
- Using $\exp(t_*)$ makes sure that the width and height will be scaled by a non-zero value ($e^0 = 1$).
- When the transform is all 0s, the output = the original proposal
- L2 regularization encourages leaving the proposal unchanged since it will force the weights of the bounding box regression head towards zero which will also force the outputs of the bounding box regression head towards zero.
- This is scale/translation invariant. The transformation specifies a relative difference between the proposal and the output. The CNN doesn't see the absolute size or position after the initial region cropping, so this allows the model to generalize for different image dimensions.

**Training bbox regression:**
- Given the proposal and the target output, you can solve for the transformation the network should have outputted to transform the proposal into the target.
$t_{x}=\left(b_{x}-p_{x}\right) / p_{w}$
$t_{y}=\left(b_{y}-p_{y}\right) / p_{h}$
$t_{w}=\log \left(b_{w} / p_{w}\right)$
$t_{h}=\log \left(b_{h} / p_{h}\right)$

## Problems with R-CNN:
1. It is very slow. You have to run a CNN on every region of interest proposal (if you have 2000 region proposals, this is 2000 forward passes through CNN per image).
2. The hand-crafted mechanism for region proposal might not be great

R-CNN is actually called Slow R-CNN because of how slow it is.

## Training R-CNN
![[gt-and-region-proposal-bbox.png]]
You receive an RGB image as input. Then, you run a region proposal algorithm to get the proposed regions. These are shown in bright blue. The ground truth labels are shown in bright green.

![[positive-neutral-negative-bbox.png]]
Next, you categorize each region as **positive, neutral, or negative**. 
- Positive: > 0.5 loU with a GT box
- Negative: < 0.3 loU with all GT boxes
- Neutral: between 0.3 and 0.5 loU with GT boxes

Positive means that the bounding box overlaps sufficiently with the ground truth bounding box. Negative means that the bounding box doesn't overlap sufficiently with a GT bbox (we need negative so the model knows what non-objects look like). Neutral is somewhere in between - not quite positive and not quite negative (like the dog's face).

![[crop-positive-neg-proposals.png]]
Then, you take the positive and negative bounding boxes (you don't use the neutral ones because you don't want to train on ambigious data) and crop and resize them to the desired dimensions. 

![[rcnn-class-and-bbox-prediction.png]]
You then run each region through the CNN. For the positive boxes you predict the class and box offset (bounding box regression transform). For the negative boxes you just predict background class (it doesn't make sense to predict a bounding box). This is basically image classification at this point.

The target box transformation we want to predict is the transformation that will turn the proposed bounding box into the ground truth bounding box. We only have a regression loss for the positive regions.

You need to use a similar region proposal at training and test time to make sure the bounding box regression works correctly
