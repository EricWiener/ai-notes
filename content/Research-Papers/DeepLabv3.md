---
tags: [flashcards]
aliases: [Rethinking Atrous Convolution for Semantic Image Segmentation]
source: https://arxiv.org/abs/1706.05587
summary: augment the ASPP module from DeepLab and remove CRF post-processing.
---
[Helpful YouTube Video](https://youtu.be/MU_KmowxkMQ)

DeepLabv3 improves on DeepLabv2 by augmenting the previously proposed [[DeepLab#Atrous Spatial Pyramid Pooling (ASPP)|Atrous Spatial Pyramid Pooling]] module and removing the [[DeepLab#Conditional Random Fields (CRFs)|CRF]]. The improvement mainly comes from including and fine-tuning batch normalization parameters in the proposed models and having a better way to encode multi-scale context via image-level features in the ASPP module.

> [!question] What does feature response density refer to?
> This refers to the `output_stride` used. When `output_stride=8`, the resulting feature map will have a higher density than when `output_stride=16`.

## Architectures to capture multi-scale context
There are four main approaches to detecting objects at multiple scales:
1. Image Pyramid: you apply a CNN to multiple scales of an input image to extract features where objects at different scales become prominent at different feature maps.
2. Encoder-Decoder: you use features at different scales from the encoder in the decoder.
3. Extra Modules: Add extra modules to capture long range information (from pixels further apart).
4. Spatial Pyramid Pooling: apply convolutions with multiple dilation rates to the same input features to process the features with multiple fields of views.

### Image Pyramid
![[Research-Papers/deeplabv3-srcs/image-pyramid.png|300]]
The same model, typically with shared weights, is applied to multi-scale inputs. The features produced from the smaller input images capture a larger FOV (better for bigger objects) and the features produced from the larger input images each have a smaller FOV (better for small objects).

The main drawback of this approach is it doesn't scale well to larger CNNs since you have to re-apply the CNN multiple times to a single input image.

### Encoder-Decoder
![[encoder-decoder.png|300]]
**Encoder**: the spatial dimension of the feature maps is gradually reduced to capture longer range information (larger effective FOV) in the deeper encoder outputs.
**Decoder**: object details and spatial dimension are gradually recovered (ex. you can increase the spatial resolution with [[Transposed Convolution|deconvolution]]).

[[Encoder-Decoder Models|U-Net]] is an example of an encoder-decoder.

### Additional Layers (aka Context Module)
You can add additional layers to the original network to capture long range information. For example, you could add a CRF to capture relationships between pixels far away from each other.

### Spatial Pyramid Pooling
![[spatial-pyramid-pooling.png|300]]
Use spatial pyramid pooling to capture context at several FOVs. [[DeepLab]] proposes [[DeepLab#Atrous Spatial Pyramid Pooling (ASPP)]], where parallel atrous convolution layers with different rates capture multi-scale information.

# Approach
Duplicate several copies of the original last ResNet block (which itself consists of three `Bottleneck` blocks) and add them in sequence to the original ResNet. They also add an ASPP module.

> [!NOTE] Naming of blocks in DeepLab vs. layers in ResNet
> ResNet starts counting its layers starting with the stem as `layer1` (or `conv1` in the table below). DeepLab starts counting its blocks after the stem, so DeepLab's `block4` refers to ResNet's `layer5`/`conv5_x`.

### Output Stride
Output stride is defined as `output_stride = input_resolution / final_resolution`. 

For image classification CNNs like ResNet, the final feature responses (before FC and global pooling) are 32 times smaller in both the height and width dimensions than the original image. Therefore, the `output_stride = 32`.

![[performance-at-different-output-strides.png]]
Using `output_stride=8` leads to better performance at the cost of more memory usage. To achieve `output_stride=8`, the last two layers of ResNet are modified to have a dilation rate of 2 and 4 respectively.

ResNet `output_stride=8` backbone ([source](https://github.com/fregu856/deeplabv3/blob/812649504118da37f3d33d760c62905d2a96c6bc/model/resnet.py#L191)):
```python
# remove fully connected layer, avg pool, layer4 and layer5:
self.resnet = nn.Sequential(*list(resnet.children())[:-4])
self.layer4 = make_layer(..., stride=1, dilation=2)
self.layer5 = make_layer(..., stride=1, dilation=4)
```

ResNet `output_stride=16` backbone ([source](https://github.com/fregu856/deeplabv3/blob/812649504118da37f3d33d760c62905d2a96c6bc/model/resnet.py#L109)):
```python
# remove fully connected layer, avg pool and layer5:
self.resnet = nn.Sequential(*list(resnet.children())[:-3])
self.layer5 = make_layer(..., stride=1, dilation=2)
```
### Adding additional layers
![[adding-additional-layers-figure.png]]
![[deeplabv3-table-2-deeper-w-atrous-convolution.png]]
- Network structures `block4`, `block5`, `block6`, and `block7` add extra 0, 1, 2, 3 cascaded modules respectively.
- Performance is generally improved by adding more blocks.

### Multi-Grid
![[resnet-architectures.png]]
DeepLabv3 tries using different dilation rates for the three `Bottleneck` blocks that are present in each of layers `conv5_x`, `conv6_x`, `conv7_x`, and `conv8_x` (`conv6_x` - `conv8_x` are all copies of `conv5_x`). 

These are denoted $\text{Multi\_Grid} = (r_1, r_2, r_3)$ as the **unit dilation rates**. Ex: (1, 2, 4) means the `conv5_1` has a unit dilation rate of 1, `conv5_2` has a unit dilation rate of 2, `conv5_3` has a unit dilation rate of 4.

The **final dilation rate** for each convulational layer is equal to the multiplication of the unit rate and the corresponding rate for that layer. For example, when `output_stride=16` and $\text{Multi\_Grid} = (1, 2, 4)$ the $3 \times 3$ `nn.Conv2D` layers in the three `Bottleneck` blocks will have rates  $\text{rates}=2 \cdot(1,2,4)=(2,4,8)$ in `conv5`.

# Improved ASPP
The main differences from DeepLab are using [[Research Papers/Batch Normalization|Batch Normalization]] and image-level features.

### Issues with large distillation rates
**Problem**
The paper found a practical issue when they tried using distillation rates that were close to the feature map size with a 3x3 kernel (although the problem applies to any kernel greater than $1 \times 1$). This is because only the center value of the kernel would come from the feature map and the other values would all be coming from zero-padding. This would result in the 3x3 filter essentially becoming a 1x1 filter where only the center value is relevant.

> [!NOTE] 
> As the sampling rate becomes larger, the number of valid filter weights (i.e., the weights that are applied to the valid feature region, instead of padded zeros) becomes smaller.

**Solution**:
They apply global average pooling to the last feature map of the model, feed the resulting image-level features to a $1 \times 1$ convolution with 256 filters (and batch norm) and then bilinearly upsample the feature to the desired spatial resolution. They then combine these features with the outputs of the other convolutions in the ASPP head and this provides global image information (which replaces the need for having large dilation rates to get a large FOV).

### Improved ASPP Implementation
 ![[improved-aspp-code.png|600]] ![[improved-aspp-diagram.png|300]]
The improved ASPP consists of (a) one 1×1 convolution and three 3 × 3 convolutions with rates = (6, 12, 18) when output stride = 16 (all with 256 filters and batch normalization), and (b) the image-level features. Note that the rates are doubled when output stride = 8. The resulting features from all the branches are then concatenated and pass through another 1 × 1 convolution (also with 256 filters and batch normalization) before the final 1 × 1 convolution which generates the final logits.

# Comparing to ResNet
### ResNet-50 with different output strides
DeepLabv3 compares their model against a ResNet-50 modified to have additional blocks (extra block5, block6, and block7 as replicas of block4. Note that in the code this would be `layer6`, `layer7`, and `layer8` as replicas of `layer5`). They experiment with different output strides up to `output_stride = 256` (no dilated convolutions at all).

A ResNet modified to have up to a block7 would have `output_stride = 256` and would have an additional `layer6`, `layer7`, and `layer8`:
![[resnet-code.png]]

![[deeplab3-table-1-resnet-50-diff-output-strides.png]]
Table 1. Going deeper with atrous convolution when employing ResNet-50 with block7 and different output stride. **Adopting output stride = 8 leads to better performance at the cost of more memory usage.**

> [!QUESTION] Not sure what the shapes of the outputs are of the extra blocks since (input size = 224) / (stride = 256) < 1

### Additional Layers with ResNet-50 vs. ResNet-101
They experiment with both ResNet-50 and ResNet-101 and see what performance impacts (measured in [[IoU|mIOU]]) adding additional blocks with a constant `output_stride=16`.

![[deeplabv3-deeper-w-resnet-50-and-101.png]]
Going deeper with atrous convolution when employ- ing ResNet-50 and ResNet-101 with different number of cas- caded blocks at output stride = 16. Network structures `block4`, `block5`, `block6`, and `block7` add extra 0, 1, 2, 3 cascaded modules respectively. 

> [!NOTE] Performance generally improves by adding more blocks with `output_stride` fixed at 16, but the margin of improvement becomes smaller.
> Also note performance peaks at `block6` for ResNet-50.

### Multi-Grid with ResNet-101
They experimented with different dilation rates for the 3x3 `nn.Conv2D` in the three `Bottleneck` blocks that are present in each of `block4`, `block5`, `block6`, and `block7`. They used a ResNet-101 with `output_stride=16`. The same $\text{Multi\_Grid}$ values were used in each block.

![[deeplabv3-table-3-multi-grid-resnet-101.png]]
 - Applying multi-grid method is generally better than the vanilla version where (r1, r2, r3) = (1, 1, 1).
 - Simply doubling the unit rates (i.e., (r1, r2, r3) = (2, 2, 2)) is not effective.
 - Going deeper with multi-grid improves the performance. Our best model is the case where block7 and (r1, r2, r3) = (1, 2, 1) are employed.

> [!NOTE] **Best results were seen with rates (1, 2, 1) with a ResNet-101 with additional layers up to `block7` (conv8)**

### Inference Strategy
They looked at the inference performance for a ResNet-101 with additional layers to `block7` and $\text{Multi\_Grid} = (1, 2, 1)$. The model is trained with `output_stride=16`.
![[deeplabv3-table-4.png]]
- When taking the model trained with `output_stride=16` and using `output_stride=8` during inference (the weights remain the same - just the dilation rate changes), you get a more detailed feature map and performance improves.
- Using multi-scale inputs (scales = {0.5, 0.75, 1.90, 1.25, 1.5, 1,75}) and left-right flipped images also improved performnace. The final result of the model was computed as the average probability from each scale and flipped image.

### ResNet-101 with Multi-Grid in `block4` and different ASPP dilation rates
For this experiment they took a ResNet-101 with `output_stride=16` and modified the $\text{Multi\_Grid}$ values as well as the dilation rates inside the ASPP module (ex. $\text{ASPP} = (6, 12, 18)$ means use rates (6, 12, 18) for the three parallel $3 \times 3$ convolution branches).
![[deeplabv3-table-5.png]]
- $\text{Multi\_Grid} = (1, 2, 4)$ performs the best.
- Adding an additional branch to the ASPP module with a dilation rate of 24 results in worse performance.
- Using image-level features improves performance.

> [!NOTE] The best model has $\text{Multi\_Grid} = (1, 2, 4)$, ASPP = (6, 12, 18), and image pooling. This is DeepLabv3


# Training Notes
### Upsampling logits vs. downsampling ground truth:
In [[DeepLab]], the target groundtruths are downsampled by 8 during training when output stride = 8. They got better performance keep the groundtruths intact and instead upsampling the final logits, since downsampling the groundtruths removes the fine annotations resulting in no back-propagation of details.

### Crop Size
The images from Pascal VOC 2012 are larger than ImageNet (ex. ImageNet is 224 x 224 and Pascal VOC 2012 can be a variety of shapes like 793 x 1123). DeepLab crops an image to a fixed size and uses this for both training and testing.

In order to avoid issues with filters with large dilation rates effectively becoming 1x1 kernels by being applied to mostly 0-padded regions, they use a large crop size of 513.

### Batch Normalization
DeepLabv3 uses batch normalization for the additional modules added to ResNet. To train batch normalization parameters a large batch size is needed.

They first train with a larger `output_stride = 16` and batch size of 16. They then freeze the batch normalization parameters and use an `output_stride=8` and train for another 30k iterations. Using a larger `output_stride` uses less GPU memory and is faster since the intermediate feature maps with `output_stride=16` are four times smaller (2 * 2) than the feature maps at `output_stride=8`. However, having the final model trained at `output_stride=8` allows producing less coarse feature maps which results in better segmentation results.

You are able to use different dilation rates withpout needing extra model parameters since you just change the spacing between the pixels the filters are applied to without requiring a larger filter.