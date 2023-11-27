---
tags:
  - flashcards
source: https://ar5iv.labs.arxiv.org/html/1706.03762
summary: the positional encodings ensure that each token has an unique value and the model can learn to attend to relative positions.
aliases:
  - Positional Encodings
---
The positional encodings used by the original [[Attention is All You Need]] paper used sine and cosine functions of different frequencies to encode the absolute position of a token in text. 

# Original paper details
In order to inform the model about the order of the sequence, the paper adds information about the position of the tokens in the sequence. The positional encodings have the same dimension $d_{\text{model}}$ as the embeddings so the two can be added together. 

[[Attention is All You Need]] uses sine and cosine functions of different frequencies:
$$\begin{array}{r}
P E_{(p o s, 2 i)}=\sin \left(p o s / 10000^{2 i / d_{\mathrm{model}}}\right) \\
P E_{(p o s, 2 i+1)}=\cos \left(p o s / 10000^{2 i / d_{\mathrm{model}}}\right)
\end{array}$$
- $pos$ is the position of the token in the sequence
- $i$ is the dimension in the token $i \in [0, d_{\text{model}})$.

### Code
```python
def get_position_angle_vec(position):
    return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
````
[Source](https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/ecfe240e66071c8179f19e32e193d1fabd16ee08/transformer/Models.py#L35). Note that `2 * (hid_j // 2)` is done because for odd dimensions (expressed as $2i + 1$), you use $2i$ in the numerator of $10000^{2 i / d_{\mathrm{model}}}$. An equivalent formulation of `get_position_angle_vec` would just subtract `1` from `i` if it was odd:
```python
def get_position_angle_vec(position):
    return [position / np.power(10000, (i - 1 if i % 2 == 1 else i) / dimensions) for i in range(dimensions)]
```


# Explanation
A clearer (and equivalent formulation) of the original equation (given by [this blog](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)) is:
$${\overrightarrow{p_t}}^{(i)}=f(t)^{(i)}:= \begin{cases}\sin \left(\omega_k \cdot t\right), & \text { if } i=2 k \\ \cos \left(\omega_k \cdot t\right), & \text { if } i=2 k+1\end{cases}$$
- $t$ is the desired position in the input sentence.
- $\overrightarrow{p_t} \in \mathbb{R}^d$ is the encoding for token $t$
- $d$ is the dimension of the encoding (note $d$ must be divisble by 2)
- $f: \mathbb{N} \rightarrow \mathbb{R}^d$ is the function that produces the output vector $\overrightarrow{p_t}$ and is a function of both the token number $t$ and the dimension $i$.
- $\omega_k=\frac{1}{10000^{2 k / d}}$

You can also imagine the positional embedding $\overrightarrow{p_t}$ as a vector containing pairs of sines and cosines for each frequency (Note that $d$ is divisble by 2):
$$\overrightarrow{p_t}=\left[\begin{array}{c}
\sin \left(\omega_1 \cdot t\right) \\
\cos \left(\omega_1 \cdot t\right) \\
\sin \left(\omega_2 \cdot t\right) \\
\cos \left(\omega_2 \cdot t\right) \\
\vdots \\
\sin \left(\omega_{d / 2} \cdot t\right) \\
\cos \left(\omega_{d / 2} \cdot t\right)
\end{array}\right]_{d \times 1}$$
where $\sin$ is used if the dimension is even and $\cos$ is used if the dimension is odd. Alternating between sin and cos is needed to ensure a linear transformation discussed below can exist.

The function was chosen because the authors thought:
> We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset $k$, $P E_{p o s+k}$ can be represented as a linear function of $P E_{p o s}$.

This means for every sine-cosine pair corresponding to frequency $\omega_k$ (a fixed dimension), there is a linear transformation $M \in \mathbb{R}^{2 \times 2}$ (that doesn't depend on $t$ - the token index) where the following holds:
$$M .\left[\begin{array}{l}
\sin \left(\omega_k \cdot t\right) \\
\cos \left(\omega_k \cdot t\right)
\end{array}\right]=\left[\begin{array}{l}
\sin \left(\omega_k \cdot(t+\phi)\right) \\
\cos \left(\omega_k \cdot(t+\phi)\right)
\end{array}\right]$$
where $$M_{\phi, k}=\left[\begin{array}{cc}
\cos \left(\omega_k \cdot \phi\right) & \sin \left(\omega_k . \phi\right) \\
-\sin \left(\omega_k . \phi\right) & \cos \left(\omega_k . \phi\right)
\end{array}\right]$$See the blog for more details. 

> [!NOTE] For a fixed dimension $k$, there is a linear relationship between a token $t$ and a token $\phi$ timesteps later ($t + \phi$).
> The linear function doesn't depend on the token index $t$ and just depends on the dimension in the token and the distance between the two tokens. This is important for learning relationships between ==relative positions==.
<!--SR:!2023-12-06,221,290-->

### Why are both sine and cosine used?
> Personally, I think, only by using both sine and cosine, we can express the sine(x+k) and cosine(x+k) as a linear transformation of sin(x) and cos(x). It seems that you can’t do the same thing with the single sine or cosine. If you can find a linear transformation for a single sine/cosine, please let me know in the comments section.

### Additional Info
\The wavelengths (how long a wave is - not the amplitude) range from $2\pi$ to $10000 \cdot 2 \pi$. The graph below shows that across the tokens for a specific dimension in a token (a row), there is a consistent wavelength.

As the dimension number increases, the wavelength grows. This makes sure that there is an unique encoding for each token (each column is unique).
![[attention-is-all-you-need-positional-encodings-20230111140723823.png]]
[[Code to generate positional encoding plot|Source code]]. [Website](https://dsalaj.com/2021/03/02/all-about-positional-encoding.html)

Note that while the top row looks like it's all 1s, it's actually still a wave. Below is a plot of just the top row's values (y-axis) and the token index (x-axis):
![[attention-is-all-you-need-positional-encodings-20230111145951727.png]]

