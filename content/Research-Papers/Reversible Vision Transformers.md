---
tags: [flashcards]
source: [[Mangalam_Reversible_Vision_Transformers_CVPR_2022_paper.pdf]]
summary: decouple number of transformer layers from the GPU memory footprint to enable training deeper models on the same hardware.
---

# Summary
[Paper Talk](https://youtu.be/AWu-f71C4Nk)
- A memory efficient architecture design for visual recognition. 
- By decoupling the GPU memory footprint from the depth of the model, Reversible Vision Transformers enable memory efficient scaling of transformer architectures. 
- The activations are efficiently re-computed vs. caching them in the GPU.
- We adapt two popular models, namely Vision Transformer and Multiscale Vision Transformers, to reversible variants and benchmark extensively across both model sizes and tasks of image classification, object detection and video classification. 
- Reversible Vision Transformers achieve a **reduced memory footprint of up to 15.5× at identical model complexity, parameters and accuracy**, demonstrating the promise of reversible vision transformers as an efficient backbone for resource limited training regimes. 
- Finally, we find that for deeper models training isn't slowed down since you can increase the batch size (since less memory is used), where throughput can increase up to 3.9× over their non-reversible counterparts.

# Introduction
- Bigger networks need increased compute and increased memory bandwidth. The compute has been increasing faster than bandwidth. The peak FLOPs has been increasing at a rate of ~3.1x every 2 years while peak bandwidth scales at a rate of ~1.4x every 2 years.
- Transformers have been doubling in compute roughly every three months for the past three years. They will/have hit a point where performance and training speed has become tightly memory bound.
- Reversible structures have a stronger inherent regularization and therefore, the paper uses a lighter augmentation recipe (repeated augmentation, augmentation magnitude and stochastic depth) and lateral connections between residual blocks.
- The memory requirement of transformers scales linearly with the depth of the model. This requires larger models to train with a smaller batch size.
- This paper was not the first to introduce a reversible architecture, but the previous strategies performed poorly when applied to vision transformers.

# Notation
$(g \circ f)(x)$ means $g(f(x))$

# Traditional Backprop
With traditional [[Backpropagation|backprop]], you will need the intermediate activations to be available. For example, consider the simplest neural network layer $f(x) = W^TX$ where $X$ is an intermediate activation inside the network. Using backprop to calculate the derivative with respect to the parent nodes and using the output $Y$ as the sole child node, you get:
$$\frac{d \mathcal{L}}{d W}=\left(\frac{d \mathcal{L}}{d Y}\right) X^T \quad \frac{d \mathcal{L}}{d X}=W \frac{d \mathcal{L}}{d Y}$$
where $\mathcal{L}$ is the loss and $Y$ is the output of $W^TX$. Note that in order to calculate the gradient of the loss with respect to the weights **you need to know the value of $X$ which is an intermediate activation**.

Keeping track of the intermediate activations is typically achieved by caching them on GPU memory for the backward pass to allow for fast gradient computation at the cost of extra memory. Furthermore, the sequential nature of the network requires the activations for all the layers to be cached in before the loss gradients are calculated and the cached memory is freed.

> [!INFO] Vanilla networks require caching activations
> The memory usage increases linearly with the depth of the network.

# Reversible Block Structure
- The reversible transformer is composed of a stack of reversible blocks that follow the structure of the reversible transformation to allow analytic invertibility of outputs.

### Reversible Transformation
The paper states:
> Consider a transformation $T_1$ that transforms an input tensor I partitioned into two $d$ dimensional tensors, $\left[I_1 ; I_2\right]$ into the output tensor $O$ also similarly partitioned into tensors, $\left[O_1 ; O_2\right]$ with an arbitrary differentiable function $F(\cdot): \mathbb{R}^d \rightarrow \mathbb{R}^d$ as follows:
>
> $$\mathbf{I}=\left[\begin{array}{c}
I_1 \\
I_2
\end{array}\right] \underset{T_1}{\longrightarrow}\left[\begin{array}{l}
O_1 \\
O_2
\end{array}\right]=\left[\begin{array}{c}
I_1 \\
I_2+F\left(I_1\right)
\end{array}\right]=\mathbf{O}$$
> Note that the above transformation $T_1$ allows an inverse transformation $T_1′$ such that $T_1^{\prime} \circ T_1$ is an identity transform.

The reason that $T_1$ has an inverse is because the inverse is given by $(O_1,O_2)\mapsto(O_1,O_2-F(O_1))$. This is because we have $I_1=O_1$, so $O_2=I_2+F(I_1)$ yields $I_2=O_2-F(I_1)=O_2-F(O_1)$. **Therefore, you can recover the input from the output and the inverse must exist.**
[Math Stack Exchange Question](https://math.stackexchange.com/questions/4552853/does-a-differentiable-function-necessarily-have-an-inverse)

### Composition of Reversible Transformations
If you have transformation $T_1$:
$$\mathbf{I}=\left[\begin{array}{c}
I_1 \\
I_2
\end{array}\right] \underset{T_1}{\longrightarrow}\left[\begin{array}{l}
O_1 \\
O_2
\end{array}\right]=\left[\begin{array}{c}
I_1 \\
I_2+F\left(I_1\right)
\end{array}\right]=\mathbf{O}$$
and transformation $T_2$:
$$\mathbf{I}=\left[\begin{array}{l}
I_1 \\
I_2
\end{array}\right] \underset{T_2}{\longrightarrow}\left[\begin{array}{l}
O_1 \\
O_2
\end{array}\right]=\left[\begin{array}{c}
I_1+G\left(I_2\right) \\
I_2
\end{array}\right]=\mathbf{O}$$
Then the composition of the functions $T=T_2 \circ T_1$ is:
$$\mathbf{I}=\left[\begin{array}{c}
I_1 \\
I_2
\end{array}\right] \underset{T}{\longrightarrow}\left[\begin{array}{c}
O_1 \\
O_2
\end{array}\right]=\left[\begin{array}{c}
I_1+G\left(I_2+F\left(I_1\right)\right) \\
I_2+F\left(I_1\right)
\end{array}\right]=\mathbf{O}$$

and this will have an inverse transform $T^{\prime}=T_1^{\prime} \circ T_2^{\prime}$. $T'$ will query $F$ and $G$ exactly once without explicitly inverting them and therefore **has the same computational cost as the forward transform $T$**.

### Learning without caching activations
> [!NOTE] An input transformed with a reversible transformation $T$ allows recalculating the input from the output of the transformation.
> Therefore a network composed of reversible transformations does not need to store intermediate activations since they can be recomputed in the backward pass from the output.

**Note**: the feature dimensions need to remain constant under $T$ for the function to be reversible (can't change the dimensions). This isn't an issue for vision transformers but is an issue for other networks that change the feature dimension (like [[ResNet]]).

# Reversible ViT
### Diagrams (for reference)
**Traditional [[ViT An Image is Worth 16x16 Words Transformers for Image Recognition at Scale|ViT]]**:
![[vit-architecture.png]]

**[[Attention#Example: CNN with Self-Attention|Self-Attention Layer]]**:
![[self-attention-module.png]]

**Reversible ViT**:
![[fig-2a-reversible-vit-block-annotated.png|400]] ![[traditional-vit.png|150]]
### Modifying ViT Architecture
Input consists of two paritioned tensors $I_1$ and $I_2$ that are transformed via [[Reversible Vision Transformers#Composition of Reversible Transformations]]:
$$\mathbf{I}=\left[\begin{array}{l}
I_1 \\
I_2
\end{array}\right] \underset{T}{\longrightarrow}\left[\begin{array}{l}
O_1 \\
O_2
\end{array}\right]=\left[\begin{array}{c}
I_1+G\left(I_2+F\left(I_1\right)\right) \\
I_2+F\left(I_1\right)
\end{array}\right]=\mathbf{O}$$

- They use a two-residual-stream design where each of the inputs $I_1$ and $I_2$ maintain their own residual streams while mixing information with each other using functions $F$ (multi-headed attention block) and $G$ (MLP sublocks).
- They leave the patichification steam from [[ViT An Image is Worth 16x16 Words Transformers for Image Recognition at Scale|ViT]] as is and initialize both $I_1$ and $I_2$ identically as the patchification output activations.
- The two residual paths are fused before the final classifier head to preserve information. They layer-normalize the inputs first and then concatenate the two residual paths.

In a typical ViT you have:
```python
def forward(x):
    x = x + self.attn(self.norm1(x))
    x = x + self.mlp(self.norm2(x))
    return x
```

However in a reversible ViT you instead have:
```python
def forward(i_1, i_2):
    o_2 = i_2 + self.attn(self.norm1(i_1))
    o_1 = i_1 + self.mlp(self.norm2(o_2))
    return o_1, o_2
```

Note that you no longer have a residual connection:
- Around the MLP block (previously you added the input to the MLP block to the output of the MLP block)
- Around the attention block (previously you added the input to the attention block to the output of the attention block)

![[vit-modified-architecture.png|300]]
The removed residual connections are shown in purple and the additional residual streams are shown in red in the diagram above.17

These connections were seen to be detrimental for training deeper models and brought no benefits for shallower models so they were removed. Instead **the residual connections for each residual stream flows through the other stream** (the attention head operates on $I_1$ and has a residual connection with $I_2$ and the MLP head operates on $I_2$ with a residual connection with $I_1$).

> [!NOTE] For the first layer $I_1$ and $I_2$ will be the same, but for the following layers they will be different (the previous $O_1$ and $O_2$ are different). Therefore, the additive connections now being used are no longer adding the input of the block to the output of the block.

# Reversible [[Multiscale Vision Transformers]]
![[reversible-mvit-architecture.png]]
