---
tags:
  - flashcards
summary: 
source: https://arxiv.org/abs/2010.04159
publish: true
---
# [[Deformable DETR]]'s Version of Deformable Attention
It was inspired by [[Deformable Convolution]] and modifies the attention module to learn to focus on a small fixed set of sampling points predicted from the features of query elements. It is just slightly slower than traditional convolution under the same FLOPs. It is slower because you are accessing memory in a random order (vs. a conv layer which always accesses memory in the same order) so memory optimization and caching doesn't work great.

![[Excalidraw/annotated-deformable-attention.excalidraw.png]]


The above diagram shows a single scale feature map $x$ ($l = 1$) and 3 attention heads in green, blue, and yellow ($M = 3$). You use $K = 3$ sampling points.

**Notation**:
- $N_q$ is the number of query features ($z_q$) you have. For the encoder, this is $H \times W$ (you pass your image through a backbone that gives a downsampled feature map. You then flatten this feature map into a sequence of length $HW$. See [[Research-Papers/DETR#Transformer encoder|DETR]]). For the decoder, this is the number of objects you want to detect ($N$).
- $K$ is the number of sampling points for each query feature to attend to the feature map. It should be much less than `grid_height` * `grid_width`.
- $z_q \in \mathbb{R}^C$ is the feature vector of query element $q$. For the encoder, this could be one pixel of the input feature map $x$. For the decoder, this could be an object query.
- $\hat{p}_q \in[0,1]^2$ is the normalized coordinates of the reference point for each query element $q$. These are normalized from the reference point $p_q$. For the encoder, the reference point can be the pixel of the input feature map. For the decoder, this can be predicted from its object query embedding via a linear projection + sigmoid.
- $x^l \in \mathbb{R}^{C \times H_l \times W_l}, l=1, \ldots, L$ are the input feature maps extracted by a CNN backbone at multiple scales. In the above diagram only a single feature map is used.

### Algorithm
- You have an input query $z_q \in \mathbb{R}^C$. 
- Apply a linear projection on $z_q$ to get the sampling offsets $\Delta \boldsymbol{p}_{m q k}$ (mask $m$, query $q$, and sampling point $k$).
- Apply a linear projection on $z_q$ and then a softmax to get attention weights $A_{mqk}$. These sum to 1 for each head.
- Apply a linear projection $W_m^{\prime} x$ on the input features $x$ to get values for each of the $M$ heads.
- Retrieve the relevant points from the above values based on the sampling offsets.
- Multiply the relevant points by their corresponding weights for each head. Then sum up all the products for each head.
- Form the outputs of each head into a single concatenated vector and apply weight matrix $W$ (this is the same as applying a weight matrix $W_m$ for each head and then adding up the results).
- You then get a combined output from all attention heads.

### Math
$$\operatorname{MultiHeadAttn}\left(\boldsymbol{z}_q, \boldsymbol{x}\right)=\sum_{m=1}^M \boldsymbol{W}_m\left[\sum_{k \in \Omega_k} A_{m q k} \cdot \boldsymbol{W}_m^{\prime} \boldsymbol{x}_k\right]
$$
The above shows multi-head attention (non-deformable). 

$$
\operatorname{DeformAttn}\left(\boldsymbol{z}_q, \boldsymbol{p}_q, \boldsymbol{x}\right)=\sum_{m=1}^M \boldsymbol{W}_m\left[\sum_{k=1}^K A_{m q k} \cdot \boldsymbol{W}_m^{\prime} \boldsymbol{x}\left(\boldsymbol{p}_q+\Delta \boldsymbol{p}_{m q k}\right)\right]$$
The above is for deformable attention on a single scale feature map (multi-scale is a bit more involved).

![[AI-Notes/Attention/deformable-attention-srcs/annotated-deformable-attn-eq.excalidraw.png]]

### Efficiency:
Efficiency is calculated with
$$O\left(2 N_q C^2+\min \left(H W C^2, N_q K C^2\right)\right)$$
Which simplifies to:
- $O\left(H W C^2\right)$ in the DETR encoder ($N_q = HW$)
- $O(NKC^2)$ in the DETR decoder ($N_q = N$)

# [[Vision Transformer with Deformable Attention|DAT]] Version
The paper Vision Transformer with Deformable Attention introduces an alternative form of deformable attention that is simpler than [[Deformable DETR]]'s implementation.
![[deformable-attention-20230105115617649.png]]
Figure 1: Comparison of DAT with other Vision Transformer models and DCN ([[Deformable Convolution]]) in CNN model. The red star and the blue star denote the different queries, and masks with solid line boundaries denote the regions to which the queries attend. In a data-agnostic way: (a) [[ViT An Image is Worth 16x16 Words Transformers for Image Recognition at Scale|ViT]] adopts full attention for all queries. (b) [[Swin Transformer]] uses partitioned window attention. In a data-dependent way: (c) [[Deformable Convolution]] learns different deformed points for each query. (d) [[Vision Transformer with Deformable Attention|DAT]] learns shared deformed points for all queries.