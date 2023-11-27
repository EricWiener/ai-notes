---
tags: [flashcards]
aliases: [Diffusion Model PyTorch Implementation]
source: https://youtu.be/a4Yfz2FxXiY
summary: PyTorch implementation of a diffusion model
---

[Colab Notebook](https://colab.research.google.com/drive/1ZClq_uh1O7T8NMsRBqWmMRC62_nLdYXq?usp=sharing)

# Noise Schedule
To sample a noised image at a particular time step we use the formula:
$$
\mathbf{x}_t=\sqrt{\bar{\alpha}_t} \mathbf{x}_0+\sqrt{\left(1-\bar{\alpha}_t\right)} \epsilon \quad \text{ where } \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$
where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t=\prod_{s=1}^t\left(1-\beta_s\right)$ where $\bar{\alpha}_t$ is the product of $(1-\beta_s)$ for all steps from $s = 1$ to $s = t$ and the noise $\beta_s$ can change depending on the step number.

# Model
- Uses a U-Net. Diffusion Models inputs and outputs need to be same spatial size.
- The model uses the same weights for each input, so we need to pass information about what time step we are currently at.

# Loss
```python
def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l2_loss(noise, noise_pred)
```