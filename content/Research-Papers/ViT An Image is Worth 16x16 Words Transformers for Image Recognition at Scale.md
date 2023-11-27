---
aliases: [Vision Transformer, ViT]
source: https://ar5iv.labs.arxiv.org/html/2010.11929
summary: applies pure transformers for image recognition by dividing an image into patches
---
[Papers with Code](https://paperswithcode.com/paper/an-image-is-worth-16x16-words-transformers-1)
[Youtube Overview](https://www.youtube.com/watch?v=j6kuz_NqkG0)

> [!note] Overview
> One of the first works to apply Transformers for vision. The paper is able to use a standard Transformer with image inputs by splitting an image into patches and providing the sequence of linear embeddings of these patches as an input to a Transformer. Image patches are treated the same way as tokens (words) in an NLP application. The model is trained for image classification.

# 


# Overview
![[vit-overview-eecs498.mp4]]

![[vit-architecture.png]]
In this picture of the ViT architecture, the input image is turned into patches along with an absolute position embedding. Without the position embedding, the patches would be treated as a bag of words in the transformer, so any notion of locality among the patches would be lost. Then, an extra learnable class embedding is added for the case of the multi-class classification that the ViT was designed for. Each of the patches is flattened and put through a linear layer. Then these tokens are passed into the transformer, which is L blocks of  encoder layers that consist of layer norm, global multi-head self attention, and dense layers. Finally, the output of only the learnable class embedding is passed into a feed-forward network and outputs the logits for prediction.


# Abstract:
- When pre-trained on large amounts of data and transferred to mid/small scale image recognition benchmarks, it performs well with fewer resources.
- Doesn't perform well when trained on small amounts of data.
- Existing approaches either apply attention in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place.
- This paper uses a pure transformer applied directly to sequences of image patches

# Introduction
- [[Transformer]]'s are computational efficient and scale well. See [[Attention is All You Need]] for computation analysis.
![[Attention#Self-Attention works on Sets of Vectors]]
- Naive application of self-attention to images would require that each pixel attends to every other pixel. With ==quadratic== cost in the number of pixels, this does not scale to realistic input sizes. Existing approaches to deal with this have been complex and not scaled efficiently on modern hardware accelerators due to specialized attention patterns.
- Transformers lack some of the [[Inductive Bias|inductive bias]] inherent to CNNs, such as translation [[Invariance vs Equivariance|equivariance]] and locality, and therefore do not generalize well when trained on insufficient amounts of data. Note: Professor Justin Johnson doesn't buy this explanation.
- Paper follows the original [[Transformer]] as closely as possible. An advantage of this intentionally simple setup is that scalable NLP Transformer architectures – and their efficient implementations – can be used almost out of the box.

# Related Work
- **ResNets have been dominating the vision field right now. People already tried to use attention in CNNs, but they didn't have as good results.**
- Multiple works try combining CNN-like architectures with self-attention (Wang et al., 2018; Carion et al., 2020)
- Some replacing the convolutions entirely (Ramachandran et al., 2019; Wang et al., 2020a).
- Parmar et al. (2018) applied the self-attention only in local neighborhoods for each query pixel instead of globally.
- Sparse Transformers (Child et al., 2019) employ scalable approximations to global self- attention in order to be applicable to images.
- Many of these specialized attention architectures demonstrate promising results on computer vision tasks, but require complex engineering to be implemented efficiently on hardware accelerators.
- The model of Cordonnier et al. (2020), which extracts patches of size 2 × 2 from the input image and applies full self-attention on top. This model is very similar to ViT, but our work goes further to demonstrate that large scale pre-training makes vanilla transformers competitive. We handle medium-resolution images as well.
- Lot of work in **combining convolutional neural networks (CNNs) with forms of self-attention**, e.g. by augmenting feature maps for image classification (Bello et al., 2019) or by further processing the output of a CNN using self-attention, e.g. for object detection (Hu et al., 2018; Carion et al., 2020), video processing (Wang et al., 2018; Sun et al., 2019), image classification (Wu et al., 2020), unsupervised object discovery (Locatello et al., 2020), or unified text-vision tasks (Chen et al., 2020c; Lu et al., 2019; Li et al., 2019).


# Method
![[vit-architecture.png]]

### Overview (Basic)
- The input image is split into 16x16 patches.
- The patches are then flattened (14x14 -> 196x1).
- Do a linear projection of the patches (all linear projections use the same weights).
- Add a positional embedding for each of the patches (so the patches aren't just like a bag of words and their order matters). The positional embedding weights are different for each position.
- Can then send into a transformer encoder and use a multi-layer perceptron (FC net) to perform the final classification.

### Overview from [[Multiscale Vision Transformers|MViT]]
The Vision Transformer (ViT) architecture starts by dicing the input video of resolution $T \times H \times W$, where $T$ is the number of frames $H$ the height and $W$ the width, into non-overlapping patches of size $1 \times 16 \times 16$ each, followed by point-wise application of linear layer on the flattened image patches to to project them into the latent dimension, $D$, of the transformer. This is equivalent to a convolution with equal kernel size and stride of $1 \times 16 \times 16$.

Next, a positional embedding $\mathbf{E} \in \mathbb{R}^{L \times D}$ is added to each element of the projected sequence of length $L$ with dimension $D$ to encode the positional information and break permutation invariance. A learnable class embedding is appended to the projected image patches.

The resulting sequence of length of $L + 1$ is then processed sequentially by a stack of $N$ transformer blocks, each one performing attention ([[Attention#Multihead Self-Attention Layer]] )multi-layer perceptron ([[Linear|mlp]]) and layer normalization ([[Layernorm]]) operations. Considering $X$ to be the input of the block, the output of a single transformer block, $\text{Block}(X)$ is computed by

$$\begin{aligned}
X_1 &=\operatorname{MHA}(\mathrm{LN}(X))+X \\
\operatorname{Block}(X) &=\operatorname{MLP}\left(\mathrm{LN}\left(X_1\right)\right)+X_1
\end{aligned}$$
The resulting sequence after $N$ consecutive blocks is layer-normalized and the class embedding is extracted and passed through a linear layer to predict the desired output (_e.g_. class). By default, the hidden dimension of the MLP is $4D$.

### Splitting and flattening patches:
The standard Transformer operates on a ==1D sequence of token embeddings==, so the 2D image needs to be reshaped accordingly.

To handle 2D images, we reshape the image $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ into a sequence of flattened 2D patches $\mathbf{x}_{p} \in \mathbb{R}^{N \times\left(P^{2} \cdot C\right)}$, where $(H,W)$ is the resolution of the original image, $C$ is the number of channels, $(P,P)$ is the resolution of each image patch, and $N=H W / P^{2}$ is the resulting number of patches (this assumes the image can be evenly split up), which also serves as the effective ==input sequence length== for the Transformer.

**Example**:
- They take a 224x224 ImageNet image and split it into a grid of 14 pixel x 14 pixel patches. 224/14 = 16, so you end up with 16x16 patches (where the name of the paper comes from) where each patch is 14 pixels on each edge.
- Because you also have a channel dimension, each of the patches ends up with dimensions 3x14x14.
- Patches are flattened from a 2D image patch with dimensions 3x14x14 to 588.

### Linear projection of flattened patches
- Linear projection from vector of size $\mathbf{x}_{p} \in \mathbb{R}^{N \times\left(P^{2} \cdot C\right)}$ -> vector of size $\mathbf{x}_{p} \in \mathbb{R}^{N \times D}$.
- The linear projection is achieved by applying a weight matrix $\mathbf{E} \in \mathbb{R}^{\left(P^{2} \cdot C\right) \times D}$ to each of the flattened patches. Ex: $\mathbf{x}_{p}^{1} \mathbf{E}$ is a $[1, (P^2 \cdot C)] \times [(P^2 \cdot C), D]$ which becomes a $(1, D)$ vector.
- The same weight matrix is used to project all the patches to dimension $D$ (which is the vector size used by all transformer layers). These projected patches are called ==patch embeddings==.
- You end up with a matrix of the form: $[\mathbf{x}_{p}^{1} \mathbf{E} ; \mathbf{x}_{p}^{2} \mathbf{E} ; \cdots ; \mathbf{x}_{p}^{N} \mathbf{E}]$ where $\mathbf{E} \in \mathbb{R}^{\left(P^{2} \cdot C\right) \times D}$. This means each patch was transformed to be D dimensions using the same projection matrix. You then group the $N$ patches into an $N \times D$ matrix.

### Prepend Learnable Embedding
- Similarly to [[BERT]]'s `[cls]` token, they preprend a learnable embedding ($\mathbf{z}_{0}^{0}=\mathbf{x}_{\text {class }}$) to the sequence of embedded patches ($[\mathbf{x}_{p}^{1} \mathbf{E} ; \mathbf{x}_{p}^{2} \mathbf{E} ; \cdots ; \mathbf{x}_{p}^{N} \mathbf{E}]$). The state of this embedding at the output of the Transformer encoder serves to ==summarize the image representation==.
- The new embedded patch matrix looks like $\left[\mathbf{x}_{\text {class }} ; \mathbf{x}_{p}^{1} \mathbf{E} ; \mathbf{x}_{p}^{2} \mathbf{E} ; \cdots ; \mathbf{x}_{p}^{N} \mathbf{E}\right]$ with dimensions $(N + 1, D)$.

### Add positional embeddings
- This is similar to [[BERT]].
- They had a learned position embedding matrix $\mathbf{E}_{\text {pos }} \in \mathbb{R}^{(N+1) \times D}$. This had the same dimensions as the embedded patch matrix with the prepended learnable embedding (from the prior step).
- They just ==add== this position embedding to the learned embeddings matrix to form the input to the transformer.
$$\mathbf{z}_{0}=\left[\mathbf{x}_{\text {class }} ; \mathbf{x}_{p}^{1} \mathbf{E} ; \mathbf{x}_{p}^{2} \mathbf{E} ; \cdots ; \mathbf{x}_{p}^{N} \mathbf{E}\right]+\mathbf{E}_{\text {pos }}, \quad \mathbf{E} \in \mathbb{R}^{\left(P^{2} \cdot C\right) \times D}, \mathbf{E}_{\text {pos }} \in \mathbb{R}^{(N+1) \times D}$$

During pre-training and fine-tuning, the classification head uses $\mathbf{z}_{L}^{0}$ (the features from the pre-pended embedding after passing through the transformer encoder). See (4) in the math below and the MLP head in the diagram below.
![[final-mlp-head-math.png]]
![[vit-an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale-20220306114844348.png]]

**Ablations**:
- They had the worst performance with no positional embeddings and got the best results with 1-D positional embeddings.
- They also tried 2-D positional embeddings. Now you have two tables with embedding vectors. You index into the first table with the row and index into the second table with the column and concatenate these two embeddings. This approach didn't show any benefits over a 1-D embedding.

### Method so-far
So far we have:
- Split the image up into a 16x16 grid of patches
- Flattened each patch
- Linearly projected each patch to have dimension $D$
- Pre-prended a learnable embedding to summarize the image features
- Added a learned positional embedding (just adding a matrix)

![[ViT An Image is Worth 16x16 Words Transformers for Image Recognition at Scale_2022-03-06 11.30.30.excalidraw]]

![[vit-creating-inputs.png]]

We have not yet covered the transformer encoder. We will do that now.

### Transformer Encoder
![[vit-an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale-20220306114726535.png]]

      

The [[Transformer]] encoder consists of alternating layers of [[Attention#Multihead Self-Attention Layer|multiheaded self-attention]] and MLP blocks (Eq. 2, 3). [[Layernorm]] (LN) is applied before every block, and [[ResNet#Residual Networks|residual connections]] after every block. The MLP contains two layers with a [[Gaussian Error Linear Unit (GeLU)|GELU]] non-linearity. The full math is below:

![[vit-full-model-math.png]]
- MSA: multiheaded self-attention
- MLP: fully connected network (multi-layer perceptron) with two layers. In the Hugging Face implementation, `intermediate_size` is the size of the first FC layer and `hidden_size` is the size of the second FC layer.
- LN: Layernorm
- Equations (2) and (3) repeat for the $L$ layers of the transformer.

### ViT Transformer Code
```python
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py#L14
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py#L67
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # Note: normalization is computed before attention
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
```

### Inductive Bias
- Vision Transformer has much less image-specific inductive bias than CNNs
- CNNs have locality, two-dimensional neighborhood structure, and translation [[Invariance vs Equivariance|equivariance]] (if you shift input, the output is shifted equally).
- In ViT, only the MLP layers are ==local and translationally equivariant==.
    - The MLP layers are really 1x1 convolutions (2D convolutions), so they have the same properties of CNNs except they don't take two-dimensional structure into account since they are just 1x1. Therefore, they also have locality and translational equivariance.
- In ViT, the self-attention layers are ==global==.
    - Self-attention operates on all input patches at the same time (so it is global vs. local)
    - The input patches are flattened, so there is no 2D structure.
    - Self-attention treats the input as a set, so you don't have translation equivariance (you append learned positional encodings, so shifting values will result in different outputs - not necessarily the same output shifted over).
- In ViT, the only time 2D structure is accounted for is when cutting the image into patches (and later when adjusting the position embeddings for images of different resolutions). Spatial relations between the patches have to be learned from scratch.

> [!NOTE] Professor Justin Johnson doesn't buy the "inductive bias" explanation since it isn't a well defined concept and you can't measure the inductive bias for an architecture.

### Hybrid Architecture
- As an alternative implementation, the input sequence can be formed from feature maps of a CNN (vs. from the raw image).

### Fine-tuning and higher resolution
- ViT is pre-trained on large datasets and fine-tuned for downstream tasks.
- To fine-tune, the prediction head is removed and a zero-initialized $D \times K$ linear layer is added ($D$ is the dimension of the transformer output and $K$ is the number of downstream classes). Note: this just means a $D \times K$ weight matrix is used + a bias term.
- It is **often beneficial to fine-tune at higher resolution than training**. The patch sizes are left the same, so the sequence length ends up being longer. The transformer can handle any length of sequence, but the positional encodings may no longer apply. The paper uses 2D interpolation of the pre-trained position embeddings. **This is the only point at which an inductive bias about the 2D structure of the images is manually injected**.

# Experiments
- They pre-train on datasets of varying size and evaluate on different benchmark tasks.
- ViT performs well at a lower computational cost. It is able to out perform ResNets when pre-trained on [[JFT-300M]] (in 1/3 the time).
- Self-supervised ViT holds promise (but didn't do amazing).

**Qualitative Results**
Below is the input and the attended to regions of the image.
![[input-and-attended-regions.png|200]]

# Analysis
There are **still convolutions used**:
- Splitting the image into patches of size $p \times p$ and then doing a linear projection is the same as Conv2D(pxp, 3->D, stride=p).
- The MLPs in the transformer are stacks of 1x1 convs.

Translation [[Invariance vs Equivariance|equivariance/invariant]]:
- The convolutional operations are translation equivariant (moving something in the input will move it in the output)
- The attention operation is a global operation, so this is **translation invariant** (moving something doesn't cause any change).

Patchwise attention is much more memory friendly than doing pixelwise attention. If you split up a 224x224 image into a 14x14 grid of 16x16 patches, this is $14^4$ entries in the attention matrix (vs. $224^4$).

Vision transformers **make more efficient use of GPUs** because matrix-multiply (used for computing attention) is more hardware friendly than conv operations.

**Justin Johnson's take:**
- Vision transformers are an evolution, not a revolution. You can still solve the same problems as with CNNs (vs. when the field switched from pre-deep learning to deep learning).
- The main benefit of Vision Transformers is probably speed. Matrix multiply is more hardware friendly than convolution, so ViTs with the same FLOPs as CNNs can train and run much faster.

# Follow-Up Work
- While ViT requires large-scale training datasets (i.e., JFT-300M) to perform well, [[Training data-efficient image transformers & distillation through attention|DeiT]] introduces several training strategies that allow ViT to also be effective using the smaller ImageNet-1K dataset.
- The architecture is unsuitable for use as a general-purpose backbone network on dense vision tasks or when the input image resolution is high, due to its low-resolution feature maps and the quadratic increase in complexity with image size.
- [[Swin Transformer]] architecture to achieve the best speedaccuracy trade-off among these methods on image classification, even though our work focuses on general-purpose performance rather than specifically on classification.

# YouTube Notes
I didn't read much of experiments, training, and results (didn't care). The following is from a YouTube summary:
[Source](https://www.youtube.com/watch?v=j6kuz_NqkG0)

### Training
- Pre-trained on JFT (Google's proprietary image dataset)
- Trained much faster than ResNet
- They measured training in TPUv3-core-days.
    - Each TPUv3 has two cores.

### Results
- Looked at results in [[VTAB]]. Transformer out performed all the other state of the art models.
- The transfer learning for huge transformers did work.
- Also looked at how the models did on ImageNet based on how much pre-training they did. ResNets do the best when there is less data. The kernel in the ResNet looks at the local image patch. As you move deeper into the network, ResNet gets a bigger receptive field.
- In the big data regime, the biggest transformer model does the best.
    - Inductive bias that CNNs have is no longer useful for big data and we should let the model learn biases itself (aka having the CNN look at image patches via a kernel works well for low data, but **for large datasets you want the model to figure out how it wants to look at the data**).
    - Note: Professor Justin Johnson doesn't buy the "inductive bias" explanation since it isn't a well defined concept.
- The transformer performance didn't saturate, so you could try to make larger models in the future.
- Biggest model had 16 head per transformer.

![[screenshot-2022-02-24_08-11-33.png]]
- In the above diagram, the area that the model was able to attend to is shown.
- The purple line drawn shows what this curve would look like for CNNs (you linearly get a larger receptive field as the layers increase).
- The circled area is where the transformers out-perform CNNs because they can pay attention to the entire image - not just the receptive field - with fewer layers.

**Self-superivsion**:
- Tried training the model with self-supervision (masked patch prediction), but this performed worse than the fully supervised pre-training (images and their corresponding labels).
- Big transformer models are pre-trained with self-supervision in NLP (predict next word given current word).
- They corrupted 50% of the image patches. They tried to predict the mean color of the corrupted patch depending on the context. They also tried predicting down-sampled version of patch embedding (4x4 instead of 16x16).

**Scaling model**
- Tried doing depth, width, and changing patch size.
- Adding additional transformer layers gave better performance (compute also goes up).
- The smaller the patch size, they actually gained more performance.