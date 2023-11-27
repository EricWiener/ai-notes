---
tags: [flashcards]
aliases: [Deformable Convolution v2]
source: https://arxiv.org/abs/1811.11168
summary: adds a learned multiplier term to deformable convolution to allow the model to change how much it weighs certain points
---

For a regular convolution, you have:
$$\mathbf{y}\left(\mathbf{p}_0\right)=\sum_{\mathbf{p}_n \in \mathcal{R}} \mathbf{w}\left(\mathbf{p}_n\right) \cdot \mathbf{x}\left(\mathbf{p}_0+\mathbf{p}_n\right)$$
where:
- $x$ is the input feature map
- $w$ is the kernel weights
- $\mathcal{R}$ is the grid that is sampled over (defined by receptive field size and dilation). Ex. $\mathcal{R}=\{(-1,-1),(-1,0), \ldots,(0,1),(1,1)\}$ is the positions corresponding to a $3 \times 3$ kernel with dilation 1.
- $p_0$ is each location on the output feature map $y$
- $p_n$ enumerates the locations in $\mathcal{R}$

![[regular-convolution-diagram.excalidraw|700]]

For [[Deformable Convolution]] you add offsets:
$$\mathbf{y}\left(\mathbf{p}_0\right)=\sum_{\mathbf{p}_n \in \mathcal{R}} \mathbf{w}\left(\mathbf{p}_n\right) \cdot \mathbf{x}\left(\mathbf{p}_0+\mathbf{p}_n+\Delta \mathbf{p}_n\right)$$
where $\left\{\Delta \mathbf{p}_n \mid n=1, \ldots, N\right\}$ and $N = |\mathcal{R}|$ ($N$ is the number of elements in $\mathcal{R}$).

For modulated deformable convolution you now have:
$$\mathbf{y}\left(\mathbf{p}_0\right)=\sum_{\mathbf{p}_n \in \mathcal{R}} \mathbf{w}\left(\mathbf{p}_n\right) \cdot \mathbf{x}\left(\mathbf{p}_0+\mathbf{p}_n+\Delta \mathbf{p}_n\right) \cdot \Delta \mathbf{m_n}$$
where $\Delta \mathbf{m_n}$ is the learned modulation scalar and lies in the range $[0, 1]$. 

**How is modulated deformable convolution different from just applying an additional convolution layer?**
??
The modulated convolution will effectively change the weight applied by the kernel to a particular element depending on the input patch it is being applied to. A regular convolution can't change the weight that is applied based on the current input patch and instead applies the same convolution to each patch of the input.
<!--SR:!2024-04-14,308,290-->