---
tags: [flashcards, eecs498-dl4cv]
source:
summary:
---

Pooling is a form of **downsampling/subsampling.**
- **No parameters learned here**
- Reduces dimensionality while retaining significant information. This helps control overfitting.
- **Hyper-parameters:** kernel size, stride, pooling function
- **No learnable parameters**
- Operates independently on each channel, so it doesn't change the number of channels

![[max-pooling-example-diagram.png]]

An example of max pooling to reduce dimensions. This is a common setting for pooling. Since we use a 2x2 filter and stride of 2, each pooling area is non-overlapping.

**Adjusting stride vs. pooling for downsampling:**
You can reduce the dimensions of the input either by changing the **stride** or by using **pooling**. Pooling has the benefit that there are no weights that need to be learned (changing stride length only makes sense when applying a filter. Filter weights need to be learned). Pooling layers are often very fast since you don’t need to compute an inner product between the input and weights.

It also introduces **invariance** to small spatial shifts (especially with max pooling). The max operation selects the largest value, so even if the picture moves a bit, the max operation might still select the same values.

Also **max pooling** can be used to introduce a non-linearity (the same way ReLU does).

### **Dimensions in a Pooling Layer:**
**Input**: $N \times C \times H \times W$
- $N$: the number of images in the batch
- $C$: the number of layers (channels)
- $H$: the height of the image
- $W$: the width of the image

**Hyperparameters**:
- Kernel size (filter size): $K$ - always a square filter
- Pooling function (max, avg)
- Stride: $S$

**Output**: $N \times C \times H' \times W'$
- $H'$: $\lfloor (H-K + 2P)/S + 1 \rfloor$
- $W': \lfloor (W - K + 2P)/S + 1 \rfloor$
- The number of channels doesn't change

**Number of output elements:** $C_{\text{out}} * H' * W'$
- $C_{\text{out}}$ tells you the number of layers in your output
- $H', W'$ are the dimensions of each layer in your output

**Floating point operations for pooling layer:**
- This is the number of multiply-add operations
- It is the (number of output elements) * (operations per output element)
- $(C_{\text{out}} * H' * W') * (K_H * K_W)$
- There are $K_H * K_W$ operations needed to compute each output element since each output is the max of a region. You can think of taking the max over a $K_H \times K_W$ area taking $K_H * K_W$ floating point operations.
- The operations per output element was $C_{\text{in}} * K_H * K_W$ for a CONV layer, but here we don't need the $C_{\text{in}}$ because the pooling layer operates on each layer independently, so one output value only comes from one layer.

**Common Settings**
- Max pooling, kernel size = 2, stride = 2
- Max pooling, kernel size = 3, stride = 2 (AlexNet). This is not that common. Pooling areas overlap since kernel size is 3 and stride is 2.