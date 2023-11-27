---
tags: [flashcards]
source: https://arxiv.org/abs/1711.07971
summary: Use a weighted sum of all features to calculate output (vs. considering just a local neighborhood)
---

# Non-Local Neural Networks
### Motivation
- Both convolutional and recurrent operations are building blocks that process one local neighborhood at a time.
- Non-local operations are a generic family of building blocks for capturing ==long-range== dependencies.
- A non-local operation computes the response at a position as a ==weighted sum of the features at all positions==. It captures long-range dependencies directly by computing interactions between any two positions, regardless of their positional distance.
- Convolutional and recurrent operations both process a local **neighborhood**, either in space or time. Long-range dependencies can only be captured when these operations are applied repeatedly, propagating signals progressively through the data. This is **computationally inefficient and has optimization difficulties.**
- Non-local blocks were shown to increase accuracy with small additional computational cost.
- ==Self-attention== can be viewed as a form of non-local mean (this work bridges self-attention for machine translation to the more general class of non-local filtering operations that are applicable to image and video problems in computer vision)
<!--SR:!2028-04-03,1723,350!2024-03-07,491,270!2024-07-04,276,323-->

### Related Work
- Non-local means is a classical filtering algorithm that computes a weighted mean of all pixels in an image. It allows distant pixels to contribute to the filtered response at a location based on patch appearance similarity.
- Feedforward modeling for sequences (an alternative to recurrent networks): long-term dependencies are captured by the large receptive fields contributed by very deep 1-D convolutions. These feedforward models are amenable to parallelized implementations and can be more efficient than widely used recurrent models.
- A self-attention module computes the response at a position in a sequence (*e.g*., a sentence) by attending to all positions and taking their weighted average in an embedding space. It can be considered a form of the non-local mean.
- Non-locality **of the model is orthogonal to the ideas of attention/interaction/relation (*e.g*., a network can attend to a local region).

### Formulation
Non-local operation in a deep neural network is defined as:

$$
\mathbf{y}_{i}=\frac{1}{\mathcal{C}(\mathbf{x})} \sum_{\forall j} f\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right) g\left(\mathbf{x}_{j}\right)
$$

- $i$ is the index of an output position (in space, time, or spacetime) whose response is to be computed.
- $j$ is the index that enumerates all possible positions
- $x$ is the input signal (image, sequence, video, features, etc.)
- $y$ is the output signal of the same size as $x$
- $f$ is a pairwise function that computes a scalar (representing relationship such as affinity) between $i$ and all $j$. This can be something like $f\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right)=e^{\theta\left(\mathbf{x}_{i}\right)^{T} \phi\left(\mathbf{x}_{j}\right)}$  where $\theta\left(\mathbf{x}_{i}\right), \phi\left(\mathbf{x}_{j}\right)$ are embeddings (ex. computed by a 1x1 conv).
- $g$ is an unary function that computes a representation of the input signal at the position $j$
- Tried out various $f$ and $g$. Non-local models are not sensitive to these choices, indicating that the generic non-local behavior is the main reason for the observed improvements. For $g$ they stuck to a linear embedding (e.g. 1x1 conv in space or 1x1x1 conv in spacetime). For $f$ they tried out multiply similarity functions (but the functions didn’t require learned weights).

### Non-local Behavior
- Non-local behavior is due to the fact that all positions $\forall j$ are considered
- A convolutional operation only sums up the weighted input in a local neighborhood (e.g., $i-1 \leq j \leq i+1$ in a 1D case with kernel size = 3).
- A recurrent operation at time $i$ is often based only on the current and the latest time steps (e.g., $j=i$ or $i - 1$).
- Different from fully-connected layers (denoted FC):
    - A non-local block computes responses based on relationships between different locations, whereas FC uses learned weights. 
    - In other words, the relationship between $x_j$ and $x_i$ is a function of the input data in non-local layers (but it isn't in FC layers). For a FC layer, the output is calculated as $y_{j}=\sum_{i} w_{i j} x_{i} + b_j$ where the output just depends on the learned weights ($w_j, b_j$) and doesn't directly relate to any other inputs besides $x_i$. The non-local block, on the other hand, uses $f\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right)$ to calculate the relation between other input positions.
    - Furthermore, non-local blocks support inputs of variable **sizes, and maintains the corresponding size in the output**. On the contrary, an FC **layer requires a fixed-size input/output and loses positional correspondence**.
    - Non-local blocks can be added anywhere in the network unlike FC that are often used in the end.

### Non-local Block
$$
\mathbf{z}_{i}=W_{z} \mathbf{y}_{i}+\mathbf{x}_{i}
$$

- Here, $\mathbf{y}_{i}$ is the non-local function ($\mathbf{y}_{i}=\frac{1}{\mathcal{C}(\mathbf{x})} \sum_{\forall j} f\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right) g\left(\mathbf{x}_{j}\right)$)
- $+\mathbf{x}_{i}$ is a residual connection that allows the non-local block to be inserted into any pre-trained model without breaking its initial behavior (ex. if $W_z$ is initialized to zeros).

![[Screenshot_2022-02-12_at_08.37.192x.png]]

- The above is an example of a non-local block. They use the Embedded Gaussian function $f\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right)=e^{\theta\left(\mathbf{x}_{i}\right)^{T} \phi\left(\mathbf{x}_{j}\right)}$ for $f$ (which gives the relationship scalar)
    - $\theta\left(\mathbf{x}_{i}\right)=W_{\theta} \mathbf{x}_{i}$
    - $\phi\left(\mathbf{x}_{j}\right)=W_{\phi} \mathbf{x}_{j}$
    - $\mathcal{C}(\mathbf{x})=\sum_{\forall j} f\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right)$ - normalization term
    - Where $\theta\left(\mathbf{x}_{i}\right), \phi\left(\mathbf{x}_{j}\right)$ are embeddings (in this case computed by 1x1x1 conv (pointwise convolution)- referred to as a bottleneck) and the normalization term is the sum of all $f(x_i, x_j)$.
- $g$ is computed by a 1x1x1 conv and the result is then scaled by the scalar computed by $f$. The result is scaled back to the original channels (it was down-sized from 1024 → 512 for the computations) via a 1x1x1 conv (denoted $W_z$).

### Training
- Models are pre-trained on ImageNet
- Fine tune with BatchNorm enabled when it is applied.
- Initialize weights with Kaiming He initialization ([https://arxiv.org/abs/1502.01852](https://arxiv.org/abs/1502.01852))

### Results
- Model is consistently better than the C2D baseline throughout the training procedure, in both training and validation error.
- The non-local behavior is important, and it is insensitive to the instantiations. In the rest of this paper, we use the embedded Gaussian version by default. This version is easier to visualize as its softmax scores are in the range of [0, 1].
- The stage where the non-local block is added doesn’t seem to matter. They always add the non-local block before the last residual block of a stage.
- Improvement is not just because of a deeper model.
- Using inputs with more frames leads better results.
- 3D Convs can capture local dependency. Non-local operations and 3D convolutions are complementary. However, non-local blocks can be more effective than 3D convolutions alone.
- Using a spacetime non-local block (what other elements you consider for the $j$ index) is better than using just space/time only versions.

> [!note]
> Adding additional non-local blocks can result in diminishing returns.
> 

### Other
**3D Convolution Inflation:**
- You can turn a 2D Conv network into a 3D Conv network by taking a trained 2D network and inflating the kernels.
- A 2D $k \times k$ kernel can be inflated as a 3D $t \times k \times k$ kernel that spans $t$ frames.
- Each of the $t$ planes in the $t \times k \times k$ kernel is initialized by the pre-trained $k \times k$ weights and rescaled by $1/t$.
- If a video consists of a single static frame repeated in time, this initialization produces the same results as the 2D pre-trained model run on a static frame.
- You can inflate only some kernels (e.g., one kernel for every 2 residual blocks to same computation and get similar results).

# Questions
- Multi-hop dependency modeling (eg. when messages need to be delivered back and forth between distant positions). [[Multi-Hop]].
    - A: this just means that multiple frames of the video need to be looked at in order to figure something out (ex. if you need to figure out which person kicked the ball last, you need to track the ball through multiple frames).
- Inflated variant of convolutional networks (7) → going from 2D to 3D
- Multi-scale testing
- [[Conditional Random Field]] (CRF - 29, 28). In the context of deep neural networks, a CRF can be exploited to post-process semantic segmentation predictions of a network
- Self-attention (49)
- *Interaction Networks* (IN) [2, 52] were proposed recently for modeling physical systems. They operate on graphs of objects involved in pairwise interactions. Hoshen [24] presented the more efficient Vertex Attention IN (VAIN) in the context of multi-agent predictive modeling.