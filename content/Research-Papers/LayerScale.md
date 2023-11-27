---
tags: [flashcards]
source: [[Going deeper with Image Transformers, Hugo Touvron et al., 2021.pdf]]
summary: build and optimize deeper transformer networks for image classification whose performance does not saturate early with more depth.
---

# Introduction
- Proposes an approach that is effective to improve the training of deeper architectures.
- Introduce the **LayerScale** approach: add a learnable diagonal matrix on output of each residual block that is initialized close to (but not at) 0. This layer improves the training dynamic, allowing ==training deeper high-capacity image transformers that benefit from depth==.
- Introduce **class-attention layers**: separate the transformer layers involving self-attention between patches, from class-attention layers that are devoted to extract the content of the processed patches into a single vector so that it can be fed to a linear classifier. This explicit separation avoids the contradictory objective of guiding the attention process while processing the class embedding. They refer to this new architecture as **CaiT** (Class-Attention in Image Transformers)
<!--SR:!2024-05-16,228,228-->

**Why use a diagonal matrix of small initial values?**
??
![[layerscale-diagram.png|200]]
Using small initial values is done to train closer to the identity function and let the network integrate the additional parameters progressively during the training. Using a scalar per-channel allows more diversity in the optimization than just adjusting the whole layer by a single learnable scalar as in [ReZero](https://paperswithcode.com/method/rezero)/[SkipInit](https://paperswithcode.com/method/skipinit), [Fixup](https://paperswithcode.com/method/fixup-initialization) and [T-Fixup](https://paperswithcode.com/method/t-fixup).
<!--SR:!2025-08-23,767,312-->

# LayerScale
- LayerScale significantly facilitates the convergence and improves the accuracy of image transformers at larger depths. It adds a few thousands of parameters to the network at training time (negligible w.r.t. the total number of weights).
- Goal is to increase the stability of the optimization when training transformers for image classification when increasing depth.
- Uses [[ViT An Image is Worth 16x16 Words Transformers for Image Recognition at Scale|ViT]] with the [[Training data-efficient image transformers & distillation through attention|DeiT]] optimization procedure. In [[ViT An Image is Worth 16x16 Words Transformers for Image Recognition at Scale|ViT]], deeper architectures have low performance and [[Training data-efficient image transformers & distillation through attention|DeiT]] only considers transformers with 12 blocks of layers (and experiments by this paper show it doesn't train deeper models effectively).
- In [[Attention is All You Need]], LayerNorm is applied after the block in the residual branch, but here LayerNorm is applied before the block (pre-norm) as advocated in "Identity mappings in deep residual networks" (Kaiming He). In the experiments, [[Training data-efficient image transformers & distillation through attention|DeiT]], doesn't converge well with post-normalization.
- In LayerScale, we initialize the diagonal with small values so that the ==initial contribution of the residual blocks== to the function implemented by the transformer is small. This will perform per-channel multiplication of the vector produced by each residual block, as opposed to a single scalar. 
- The motivation is therefore closer to that of [ReZero](https://paperswithcode.com/method/rezero), [SkipInit](https://paperswithcode.com/method/skipinit), [Fixup](https://paperswithcode.com/method/fixup-initialization) and [T-Fixup](https://paperswithcode.com/method/t-fixup): to train closer to the identity function and let the network integrate the additional parameters progressively during the training.
<!--SR:!2024-05-17,456,270-->

LayerScale is a per-channel multiplication of the vector produced by each residual block, as opposed to a single scalar. The objective is to group the updates of the weights associated with the same output channel. Formally, LayerScale is a multiplication by a diagonal matrix on output of each residual block. In other words:
$$\begin{gathered}
x_l^{\prime}=x_l+\operatorname{diag}\left(\lambda_{l, 1}, \ldots, \lambda_{l, d}\right) \times \operatorname{SA}\left(\eta\left(x_l\right)\right) \\
x_{l+1}=x_l^{\prime}+\operatorname{diag}\left(\lambda_{l, 1}^{\prime}, \ldots, \lambda_{l, d}^{\prime}\right) \times \operatorname{FFN}\left(\eta\left(x_l^{\prime}\right)\right)
\end{gathered}$$
[Source](https://paperswithcode.com/method/layerscale)

### LayerScale Analysis
- Depth is one of the main sources of instability when training Vision Transformers.
- Find [[Stochastic Depth]] is very important when training deeper models. Unlike the DeiT implementation that uses Stochastic Depth with a per-layer drop-rate that depends linearly on the layer depth, this paper finds it doesn't provide any benefit over the simpler uniform drop-rate.
- LayerScale makes the ratio between the norm of the residual activations and the norm of the activations of the main branch more uniform across layers ($\left\|g_{l}(x)\right\|_{2} /\|x\|_{2}$).

### LayerScale Code
[Source](https://github.com/rwightman/pytorch-image-models/blob/909705e7ffd8ac69ca9088dea90f4d09d0578006/timm/models/cait.py#L110)
```python
class LayerScaleBlockClassAttn(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add CA and LayerScale
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_block=ClassAttn,
            mlp_block=Mlp, init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x, x_cls):
        # x is the input and x_cls are the learnt class embeddings
        u = torch.cat((x_cls, x), dim=1)
        x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        return x_cls
```
LayerScale is implemented by multiplying by `self.gamma_1` and `self.gamma_2` which are learnable parameters that are initialized to very small values (default is 1e-4)

Example of this block being used is [here](https://github.com/rwightman/pytorch-image-models/blob/909705e7ffd8ac69ca9088dea90f4d09d0578006/timm/models/cait.py#L318)

# Class Attention
- Offers a more effective processing of the class embedding.

# Related Work
- [[ResNet]] do not offer a better representational power. They achieve better performance because they are easier to train. It is important there is a clear path both forward and backward and advocated using the identity function as the skip connection.

### Fixup, [[ReZero]], SkipInit
![[layerscale-vs-fixup,rezero,skipinit.png]]
- These all introduce learnable scalar weightning on the output of residual blocks while removing the pre-normalization and the warmup.
- Rezero, Fixup and T-Fixup do not converge when training DeiT off-the-shelf. 
- FixUp: initialize classifcation layer + last layer of each residual branch to 0; scale only weights terms in other layers during initialization; add a scalar multiplier $\alpha = 1$ at every branch; and add a scalar bias (initialized at 0) before every convolution, linear, and element-wise activation. (2019). This paper re-introduces LayerNorm and warmup in order to get FixUp and T-FixUp to converge.
- SkipInit: allow normalization-free training of neural networks by downscaling residual branches at initialization. Sets $\alpha = 0$ at initialization. (2020)
- [[ReZero]]: initializes each layer to perform the identity operation. Sets $\alpha = 0$ at initialization (2020). This paper finds it is better to set $\alpha = \epsilon$ vs. setting $\alpha = 0$. 

### Fixup
FixUp Initialization, or Fixed-Update Initialization, is an initialization method that rescales the standard initialization of residual branches by adjusting for the network architecture. Fixup aims to enables training very deep residual networks stably at a maximal learning rate without normalization.

Fixup and T-Fixup are competitive with LayerScale in the regime of a relatively low number of blocks (12–18). However, they are more complex than LayerScale: they employ different initialization rules depending of the type of layers, and they require more changes to the transformer architecture.

### SkipInit
SkipInit is a method that aims to allow normalization-free training of neural networks by downscaling residual branches at initialization. This is achieved by including a learnable scalar multiplier at the end of each residual branch, initialized to $\alpha = 0$.

The method is motivated by theoretical findings that batch normalization downscales the hidden activations on the residual branch by a factor on the order of the square root of the network depth (at initialization). Therefore, as the depth of a residual network is increased, the residual blocks are increasingly dominated by the skip connection, which drives the functions computed by residual blocks closer to the identity, preserving signal propagation and ensuring well-behaved gradients. This leads to the proposed method which can achieve this property through an initialization strategy rather than a normalization strategy.
