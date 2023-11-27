---
aliases: [CONV layer, CNN]
---
Files: 498_FA2019_lecture07.pdf
Tags: EECS 445, EECS 498 F20

Convolutional Neural Networks are commonly used for image classification. A traditional Neural Network flattens the image into a vector. This loses spatial information.

CNNs exploit the inherent structure in images. It takes an image as input and outputs the scores for each label. This replaces how NN usually use a flattened representation.

You could use a NN to achieve the same thing as a CNN, but using a CNN drastically reduces the number of filters, since you can apply sweeping filters without having to have multiple nodes for each sweep. This is also faster than a NN.

### Layers
A normal NN has fully connected layers and an activation function. A CNN will additionally need a convolution layer, pooling layer, and normalization.

# Convolution (CONV)
- Most important layer in CNN.
- Each neuron in this layer is a filter.
- The filter helps preserve spatial relationship of pixels
    - Pixels close to each other are usually more relevant.
    - Conv layers have translation [[Invariance vs Equivariance|equivariance]] (if you shift the entire input, the output of the conv layer will just be a shifted version of the previous input). This is one example of the kernel respecting spatial positions better.
- There are an infinite number of different filters that can be learned.
- You can have multiple filters applied to the same image. These would be additional layers to the output.

![https://paper-attachments.dropbox.com/s_3DF3CEF0F44773CB0740DE99F28A214BF20AB0F9D087301742DC19A7753ED5AC_1584370598840_image.png](https://paper-attachments.dropbox.com/s_3DF3CEF0F44773CB0740DE99F28A214BF20AB0F9D087301742DC19A7753ED5AC_1584370598840_image.png)

![https://paper-attachments.dropbox.com/s_3DF3CEF0F44773CB0740DE99F28A214BF20AB0F9D087301742DC19A7753ED5AC_1584370707635_Screen+Shot+2020-03-16+at+10.58.22+AM.png](https://paper-attachments.dropbox.com/s_3DF3CEF0F44773CB0740DE99F28A214BF20AB0F9D087301742DC19A7753ED5AC_1584370707635_Screen+Shot+2020-03-16+at+10.58.22+AM.png)

The filter moves over the image in strides. It applies a weighted sum of the inputs (can have a bias) to each window to create a new output image.

> [!note]
> With linear models and fully connected networks you computed the dot product between the input and the weights at once. With convolutional kernels you compute the dot product with smaller sections of the input image and move the kernel around.
> 

A **bias term** allows you to learn a linear function with offset $Wx + b$ for each filter. You have a single bias term per filter.

**Stride length:** how much you move the filter over by each time. 

The actual weights of the filter are learnt throughout the process of the algorithm. These aren’t set by you.

When you have different weights applied to each channel (red, green, blue), you still get a scalar out. It still a linear combination, but with three times as many elements.
![[single-scalar-out-from-conv.png]]

You get a single scalar out each time you apply the filter

### Applying Multiple Filters
You can apply filters multiple times. This results in a larger output

![[multiple-convolutional-layers.png]]

![[AI-Notes/Concepts/convolutional-neural-network-srcs/Screen_Shot 2.png]]

The outputs of applying the filter are sometimes called **activation maps** because they show how much each position of the input activated a certain filter.

You can also think of the output $6 \times 28 \times 28$ grid as being a $28 \times 28$ grid with similar size to the input, which has a $6$ dimensional feature vector at each entry. This feature vector tells us something about the position in the original image.

The outputs preserve the original structure of the image. The output will have as many channels as the kernel weights. Each entry in the output tells us how much each corresponding group of entries in the input responded to the kernel.

### Batching Input
It's common to send multiple images through at the same time. This means you will have multiple outputs.

![[AI-Notes/Concepts/convolutional-neural-network-srcs/Screen_Shot 3.png]]

### **Hyperparameters:**
There are multiple hyperparameters you can adjust in a CNN.
- The number of filters (number of neurons in the hidden layer)
- Filter size. Filters are usually square. The square filter is often selected just because **there's no preference in which direction a pattern can be found**.
- Padding
- Stride length

## Padding
When you apply the filter, you end up with an output with smaller dimensions (width and height) than the input. Assuming square image and filter (most kernels are square), we have:
- Input: $W$
- Filter: $K$
- Output: $W - K + 1$

> [!note]
> Problem: the feature maps shrink with each layer, which puts constraints on the depth of layers you can have
> 

**Zero padding:**

![https://paper-attachments.dropbox.com/s_3DF3CEF0F44773CB0740DE99F28A214BF20AB0F9D087301742DC19A7753ED5AC_1584371358084_Screen+Shot+2020-03-16+at+11.09.14+AM.png](https://paper-attachments.dropbox.com/s_3DF3CEF0F44773CB0740DE99F28A214BF20AB0F9D087301742DC19A7753ED5AC_1584371358084_Screen+Shot+2020-03-16+at+11.09.14+AM.png)

- Helps preserve the size of the input. Makes images appear bigger. ****
- You pad zeroes around the edges of an image to preserve its dimensions. If you didn’t have zero padding, the top left pixel would only be counted once. With padding, this pixel can be counted more often.
- This is the most common approach

### [Same padding vs. valid padding](https://stackoverflow.com/a/48393040/6942666)
**Same padding**: Apply padding to input (if needed) so that input image gets fully covered by filter and stride you specified. For stride 1, this will ensure that output image size is same as input. Set $P = (K - 1) / 2$. Now the output is given as $W - K + 1 + 2P$ (for a stride of 1).

**Valid padding:** Don't apply any padding, i.e., assume that all dimensions are valid so that input image fully gets covered by filter and stride you specified.

**Notes**
- This applies to conv layers as well as max pool layers in same way
- The term "valid" is bit of a misnomer because things don't become "invalid" if you drop part of the image. Sometime you might even want that. This should have probably be called `NO_PADDING` instead.
- The term "same" is a misnomer too because it only makes sense for stride of 1 when output dimension is same as input dimension. For stride of 2, output dimensions will be half, for example. This should have probably be called `AUTO_PADDING` instead.
- In `SAME` (i.e. auto-pad mode), Tensorflow will try to spread padding evenly on both left and right. However, if the amount of padding needed is an odd number, Tensorflow will add more padding to one side than the other.
- In `VALID` (i.e. no padding mode), Tensorflow will drop right and/or bottom cells if your filter and stride doesn't full cover input image.

### Downsampling with stride length

The **receptive field** tells you what area in the input, a certain output element depends on.

![[A  receptive field in the   previous layer   ]]

A "receptive field in the **previous layer**"

![[A  receptive field in   the input    ]]

A "receptive field in **the input**" 

It's important to note that each successive convolution adds $K - 1$ to the receptive field size. With L layers the receptive field size is $1 + L \times (K – 1)$. Ex: for $L = 2$, you have $1 + 2 \times (3 - 1) = 5$ so you have a $5 \times 5$ receptive field. 

> [!note]
> In order for a layer convolution to "see" the entire image, you need many layers for a large image. The solution to this is **downsampling.**
> 

You can adjust the **stride** to reduce the size of the output. With input dimension $W$, filter size $K$, padding $P$, and stride of $S$, the output size is $(W - K + 2P) / S + 1$

![[Ex  with a kernel of 3 and stride of 1, the output for a 7x7 input is 5x5. With a stride of 2, the output is a 3x3.]]

Ex: with a kernel of 3 and stride of 1, the output for a 7x7 input is 5x5. With a stride of 2, the output is a 3x3.

Here is a **[receptive field calculator](https://fomoro.com/research/article/receptive-field-calculator)**.

Stride can be used for down sampling and effectively can just throw away some of the information, so **your filter can see more of the image.**

This lets us get **larger receptive fields with fewer layers.**

![[The colored circles are the pixels in the input image. This shows how with a stride of 2, you will reduce the input size from 7 to 3.]]

The colored circles are the pixels in the input image. This shows how with a stride of 2, you will reduce the input size from 7 to 3.

### **Dimensions of a CONV layer:**

![[AI-Notes/Concepts/convolutional-neural-network-srcs/Screen_Shot 7.png]]

**Input**: $N \times C_{\text{in}} \times H \times W$

- $N$: number of images in the batch
- $C_{\text{in}}$: the number of layers (channels)
- $H$: the height of the image
- $W$: the width of the image

**Weight matrix**: $C_{\text{out}} \times C_{\text{in}} \times K_H \times K_W$

- Each filter has a weight for each incoming layer
    - Each incoming layer is of size $K_H \times K_W$
    - There are $C_{\text{in}}$ incoming filters
    - **The number of channels in a single filter must match** $C_{\text{in}}$
- There are $C_{\text{out}}$ filters applied to the incoming layer.
    - Each of the $C_{\text{out}}$ filters has $C_{\text{in}}$ channels.
    - The output will have $C_{\text{out}}$ channels.
- Weight matrix isn't affected by batch size.

**Bias vector**:  $C_{\text{out}} \times 1$ (just a column vector)

**Output**: $N \times C_{\text{out}} \times H' \times W'$

- $H'$: $\lfloor (H-K + 2P)/S + 1 \rfloor$
- $W': \lfloor (W - K + 2P)/S + 1 \rfloor$
- Kernel size (filter size): $K_H \times K_w$
- Number filters: $C_{\text{out}}$
- Padding: $P$.
- Stride: $S$

**Number of output elements:** $C_{\text{out}} * H' * W'$

- $C_{\text{out}}$ tells you the number of layers in your output
- $H', W'$ are the dimensions of each layer in your output

**Memory used for output (KB):**

- Number of output elements = $C_{\text{out}} * H' * W'$
- Bytes per element = 4 (for 32-bit floating point)
- KB = (number of elements) * (bytes per elem) / 1024

**Number of learnable values for a layer:**

- Weight shape: $C_{\text{out}} \times C_{\text{in}} \times K_H \times K_W$
- Bias shape: $C_{\text{out}}$
- Number of weights = $C_{\text{out}} * C_{\text{in}} * K_H * K_W + C_{\text{out}}$ (add $C_{\text{out}}$ for bias)

**Number of floating point operations in layer:**

- This is the number of multiply-add operations because lots of hardware can compute this in one cycle.
- It is the (number of output elements) * (operations per output element)
- $(C_{\text{out}} * H' * W') * (C_{\text{in}} * K_H * K_W)$
- Operations per output element is $C_{\text{in}} * K_H * K_W$ because you need to perform a dot product for $C_{\text{in}}$ filters which each take $K_H * K_W$operations.

### Common settings for CONV layers:

![[/Screenshot 2022-01-29 at 09.06.20@2x.png]]

### Example

- Input volume: 3 x 32 x 32
- 10 5x5 filters with stride 1, pad 2

**Output volume:**

- $C_{\text{out}} = 10$
- $H' = \lfloor (32-5+2*2)/1 + 1 \rfloor = 32$
- $W' = \lfloor (32-5+2*2)/1 + 1 \rfloor = 32$
- $N \times 10 \times 32 \times 32$

**The number of learnable parameters would be:**

- 10 filters
- Each filter has size $3 \times 5 \times 5$ ($3$ because of the three input channels - r,g,b)
- 10 bias terms (one per filter)
- $10 * (3 * 5 * 5) + 10 = 760$ total learnable parameters

**Number of multiply-add (floating point) operations:**

- The output size is $10 \times 32 \times 32$, which is $10,240$ total outputs.
- Each output is the inner product of two $3 \times 5 \times 5$ tensors (75 elements)
- Total: $75 * 10,240 = 768,000$

### **Efficiency Concerns**

- **Parameter sharing** just means that when we convolve with the filter, we use the same weights in the filter for each convolution. If we didn't do this, we would need a separate set of weights for each spatial (x, y) position.
    - This makes the assumption that if one feature is useful to compute at some spatial position (x,y), then it should also be useful to compute at a different position (x2,y2).
    - However, each filter has different weights and each CONV layer has different weights

### Activation Function

> [!note]
> If you stack multiple convolution filters in a row, this is just combining linear operations. You just end up with another convolution.
> 

To avoid the convolutions just being one big linear operator, we apply an activation function after a convolution. This adds a non-linearity, so we can learn a non-linear function. Otherwise, we would be stuck always learning a linear function.

**Rectified Linear Unit (ReLU)**
Element wise function to output of the CONV layer.
- No parameters learned here
- This ensures that the image will be valid and not have negative values.

### What do convolutional filters learn?
You can look at the **output of an image** after a certain filter. This lets you see what is going on in the image.

![[Visualizing the output of an   image   after passing through each kernel. This is   not   showing the kernel weights themselves. ]]

Visualizing the output of an **image** after passing through each kernel. This is **not** showing the kernel weights themselves. 

- The earlier layers tend to pick out details like oriented edges and opposing colors (ex. looking for sharp changes between colors).
- The response in the output map tells you how much the image responds to each filter
- The latter CONV layers tend to pick out more wholistic things.

### 1v1 Convolution

- Allow you to reduce the number of channel layers while preserving dimensions of the image.
- Consolidates feature maps to reduce “filter dimensionality”.
- You use no padding, $1 \times 1$ filter, and $1$ as stride.

![https://paper-attachments.dropbox.com/s_ABDC441EE38556584F7F689EBF72D7299613FFE6946369D85BBE9A283E034767_1584496962710_Screen+Shot+2020-03-17+at+10.02.39+PM.png](https://paper-attachments.dropbox.com/s_ABDC441EE38556584F7F689EBF72D7299613FFE6946369D85BBE9A283E034767_1584496962710_Screen+Shot+2020-03-17+at+10.02.39+PM.png)

In this example, you have an input thats $64 \times 56 \times 56$ and an output that is $32 \times 56 \times 56$. Each filter has size $1 \times 1 \times 64$ and performs a 64-dimensional dot product, producing one scalar. Since you have 32 of these filters, your output will have a depth of $32$.

If you go back to the interpretation of the output of a convolution as being a $56 \times 56$ grid with feature vectors of size $64$, you can think of this as a linear layer operating independently at each position in the grid (in other words - it gives a fully connected network operating on each input position).

You sometimes see Neural Network structures where you have a 1x1 CONV, followed by a ReLU, then a 1x1 CONV, ReLU, etc. This is called a **network in network structure** because it is a fully connected neural network that operates independently on each feature vector.

The benefit of a $1 \times 1$ CONV layer is that you need much fewer parameters than if you were to flatten the entire CONV output and pass it through an FC network.

![[Source](https://medium.com/analytics-vidhya/talented-mr-1x1-comprehensive-look-at-1x1-convolution-in-deep-learning-f6b355825578)](Convolutio%2092082/Untitled.png)

[Source](https://medium.com/analytics-vidhya/talented-mr-1x1-comprehensive-look-at-1x1-convolution-in-deep-learning-f6b355825578)

The above diagram shows another example of a 1x1 convolution. The input is $64 \times 64 \times 192$ (blue). You will then have an orange convolution layer of size $C_{\text{out}} \times 1 \times 1 \times 192$. This performs a dot product with each corresponding $1 \times 1 \times 192$ vector in the input which computes an element-wise product and then sums up. This performs a similar computation to a fully connected network, but you operate independently on each $1 \times 1 \times 192$ vector in the input (and do this $64 \times 64$ times). 

![[Pooling Overview]]

# Fully Connected Layer (FC)

The entire network isn’t fully connected, but some layers are. Flatten out the image and apply a fully connected layer to it.

- This is a normal feedforward NN layer.
- You need to flatten the features in order to use the fully connected layer. This loses spatial features
- Ideally, you have done the feature learning in the previous layers that are significant on their own and don’t necessarily require the spatial locality.

## Flattening Layer

- Takes Convolutional Layer output and flattens it
- Takes a dxd matrix and returns a d*d sized vector

## Output layer

Outputs the final values of your CNN, possible output:

- Final prediction (1 dimensional output)
- “Scores”/logits for each class (dimensions = num. of classes).
    - Usually you predict the highest scoring class.
- Probabilities for each class (dimensions = num. of classes)

## Training + Implementation

Similar to training a NN. You set a loss function and then update with the gradients.

Usually as you move through the network, the **spatial size decreases** (using pooling or strided convolutions), but the **number of channels increases** (total "volume" is preserved). This is common in the practice.

# Normalization

It is often hard in practice for the network to converge when trying to train it. The idea is to receive the outputs of the previous layer and normalize them to they have **zero mean and unit variance** (standard deviation = 1). This is done to reduce "**internal covariate shift**." 

Rough idea is when you train a deep neural network, each layer is dependent on the previous layer. As you train the weights in one layer, the outputs of that layer will change. The next layer will be receiving outputs that might have a changing distribution, which might be bad for optimization.

To overcome the problem, you want to standardize the output of the layers to fit some target distribution (zero mean and std dev = 1). You can do this by subtracting the mean and dividing by the standard deviation.

$$
\hat{x}^{(k)} = \frac{x^{(k)} - E[x^{(k)}]}{\sqrt{\text{Var}[x^{(k)}]}}
$$

This is a differentiable function, so we can use it as an operator in our networks and backprop through it.

### [[Batch Normalization]] (Neural Networks)

Batch normalization is one of the most common forms of normalization. You pass in $N$ vectors of length $D$. You then compute the mean and standard deviation over each channel in the vectors (ex. compute the mean + std dev for each channel across all spatial positions). **You take advantage of the entire batch to try to get representative means and standard deviations.**

![[IMG_73A45F90D804-1.jpeg.jpeg|200]]
Note that $N$ here is the batch size. This diagram is discussing batch normalization for a fully connected network - **not** a convolutional network. The red area shows a single 

Note that since you are computing the mean and standard deviation over all $N$ vectors per feature, you will end up with a mean and standard deviation vectors of length $D$. 

- The $\epsilon$ term in the denominator of the normalization is to avoid dividing by zero if the standard deviation is zero.

> [!note]
> BatchNormalization is the first time we have combined data along the batch dimension so far.
> 

Note that each BatchNorm layer’s channels will have their own $\mu$ and $\sigma$.

**Scale and shift parameters:**

Making the mean zero and the standard deviation 1 is a hard constraint to put on the network. It's possible it could be too hard of a constraint. You can introduce learnable scale and shift parameters $\gamma, \beta$ (of size $D$) that will let the network learn what mean and standard deviation it would like the incoming values to have for each layer.

Now instead of the output being $\hat{x}$, it is: $y_{i, j} = \gamma_j \hat{x}_{i, j} + \beta_j$

- Output shape is $N \times D$
- Learning $y = \sigma$ and $\gamma = \mu$ will be the same as the original function

Note that batch normalization can be fused into the previous fully-connected or conv layer at test time since it just becomes a linear operator.

**Problem: Estimates depend on minibatch, can't do this at test time**

The $\mu$ and $\sigma$ are calculated over the minibatch. This isn't possible at test time because you only have a single input. Don't want your inputs depends on each other.

During training, you can keep track of a running mean and running ****standard deviation, which will become a fixed scalar that can be used at testing. **Note:** during training you just compute $\mu$ and $\sigma$ over the mini-batch, but during test time you use the running average.

During testing, batchnorm becomes a linear operator, so it can be fused with previous fully-connected or convolutional layer (just subtracting and dividing by a constant scalar).

> [!note]
> Batch size was already affecting performance because it affected the noise in the gradient updates that we computed (backward pass). Now it will also affect the training time forward pass because of BatchNorm.
> 

> [!note]
> You can end up with a mis-match if you use a different batch size during training and testing. Want as close a match as possible between the two.
> 

> [!note]
> BatchNorm is usually inserted after FC or Conv layers and before nonlinearities, but you can put before/after the nonlinearity.
> 

**Benefits:**

- Makes deep networks much easier to train
- Allows higher learning rates, faster convergence
- Networks become more robust to initialization
- Acts as regularization during training
- Zero overhead at test-time: can be fused with CONV.

**Cons:**

- Not well understood theoretically (hand waving around "internal covariance shift").
- Behaves differently during training and testing: this is a common source of bugs
- Doesn't make sense to always normalize if your datasets are not well balanced

## [[Batch Normalization]] (CNN)

![[AI-Notes/Concepts/convolutional-neural-network-srcs/Screen_Shot 10.png]]

Previously we averaged over the $N$ vectors and then got a result that was $1 \times D$. With CNN's, you still compute over the $N$ entries in the batch, but you don't compute over the channels. You do compute over the height and width (independently of each other). You end up with a $1 \times C \times 1 \times 1$ $\mu, \sigma$. In NN we though of each feature being one dimension of the input vectors. In a CNN, we think of a feature as being a single channel.

## Layer Normalization

Batch normalization acts differently during training and testing because there is no batch to use when testing. Layer normalization is an alternate approach that computes the mean and standard deviation over the features (instead of over the batch). This means you can still do this at test time. You will end up with an $N \times 1$ output (instead of $1 \times D$).

For layer norm, you take the mean and standard deviation for every output that corresponds with a particular training image (over all the channels and over all the $x, y$).

Works well for fully connected networks. Used in RNNs, transformers.

## Instance Normalization

This is an equivalent concept of Layer Normalization for CNNs. Instead of averaging over the batch dimension, we just normalize over the spatial dimensions (over $H$ and $W$, but not $N$). You end up with an $N \times C \times 1 \times 1$ output instead of $1 \times C \times 1 \times 1$ output.

## [Normalization Comparison](https://www.youtube.com/watch?v=l_3zj6HeWUE&t=563s)

![[AI-Notes/Concepts/convolutional-neural-network-srcs/Screen_Shot 11.png]]

When looking at this diagram, it's important to note that the 2D grid for each CNN layer is unraveled. The $H \times W$ matrix becomes just a single vector of length $H * W$.

One feature corresponds to a single channel. For instance, in the first layer of the CNN, you would have a red, green, and blue feature (one feature for each channel). Each image would then have three features. It's important to note that you group all the pixels in a channel together into a feature. You don't care about their position in the image.

Every time you compute statistics, you just compute the mean and standard deviation for the entire blue highlighted section and get scalar results (not a vector of results).

![[Convolutio%2092082 IMG_73A45F90D804-1.jpeg]]

- Batch Norm: compute statistics over an entire image's feature. For instance, you would normalize all the red channel values for every image. You end up with $C$ means total (one for each feature/channel).
- Layer Norm: the statistics are computed across each feature and are independent of other examples. For instance, you would take the mean of the red, green, and blue values for a single image and normalize all of them. Layer norm doesn't depend on the size of the mini-batch, since each image is normalized separately. The idea here is that the images might be sufficiently different that they shouldn't be normalized together.
- Instance Norm: computes statistics only for one feature for a single image. This would be like computing the mean for red pixels for a single image. The idea here is that each channel in each image might be sufficiently different that they should be handled separately.
- Group Norm (relatively new): this is a middle-ground between layer norm and instance norm. You still treat every image separately, so the batch-size doesn't matter. However, you now group multiple features together when computing statistics. For example, you would group the red and the green channels together for a single image when computing statistics. The idea here is that some features might be very closely related to each other, such as multiple filters that detect edges.

# Other

### General

- As you go through Conv Networks, the **spatial size tends to decrease** (smaller height and width) using pooling or strided convolutions. The number of **channels tends to increase**. This causes the total “volume” to be preserved so you can still maintain information.

**Dealing with rectangular images:**

Most CNN stuff we have seen works with square images. If an image is a rectangle, you can resize it and then crop it.

### 1D Convolution

![[/Screenshot 2022-01-30 at 06.31.01@2x.png]]

- Inputs are $C_{\text {in }} \times W$ (only one spatial dimensions - $W$. There is no height).
- Weights are $C_{\text {out }} \times C_{\text {in }} \times K$ (slidden along the $W$ dimension of the input).
- This is used with 1 dimensional data like audio, time series, or text.

![https://paper-attachments.dropbox.com/s_3DF3CEF0F44773CB0740DE99F28A214BF20AB0F9D087301742DC19A7753ED5AC_1585169879733_Image+3-25-20+at+4.57+PM.jpg](https://paper-attachments.dropbox.com/s_3DF3CEF0F44773CB0740DE99F28A214BF20AB0F9D087301742DC19A7753ED5AC_1585169879733_Image+3-25-20+at+4.57+PM.jpg)

Can apply the same concept of filters to sections of sound in order to work with audio data.