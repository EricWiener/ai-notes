---
tags: [flashcards]
source: [[Improved Denoising Diffusion Probabilistic Models]]
summary: changes how you add the timestamp embedding to the diffusion model with GroupNorm
---

> [!NOTE]
> Instead of computing a timestamp embedding $v$ based on timestamp $t$ and adding it to the hidden state $h$ via $\text{GroupNorm}(h + v)$ you instead compute a scale $w$ and shift $b$ matrices and compute $\text{GroupNorm}(h) * (w+1) + b$.

Adaptive GroupNorm changes the way that you include a timestamp embedding into a diffusion model. The previous approach was to do:
```python
t_emb = self.emb_layers(t_emb)
feats = self.in_layers(feats)
feats = feats + t_emb
feats = nn.GroupNorm()(feats)
out = self.out_layers(feats)
```
where `GroupNorm` is typically included in `out_layers` but is shown seperately to make it easier to see the difference.

The updated approach is to do:
```python
t_emb = self.emb_layers(t_emb)
feats = self.in_layers(feats)
feats_norm = nn.GroupNorm()(feats)
feats = feats_norm * (1 + scale) + shift
out = self.out_layers(feats)
```
where you predict `scale` and `shift` matrices using `emb_layers` (`out_channels` is multiplied x2) and then add these to the already normalized features.

[Code](https://github.com/openai/improved-diffusion/blob/6051920c0865d53146db7841bfba43c3e67edef6/improved_diffusion/unet.py#L184)

```python
def __init__(...):
    self.in_layers = nn.Sequential(
        normalization(channels),
        SiLU(),
        conv_nd(dims, channels, self.out_channels, 3, padding=1),
    )
    self.emb_layers = nn.Sequential(
        SiLU(),
        linear(
            emb_channels,
            2 * self.out_channels if use_scale_shift_norm else self.out_channels,
        ),
    )
    self.out_layers = nn.Sequential(
        normalization(self.out_channels),
        SiLU(),
        nn.Dropout(p=dropout),
        zero_module(
            conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
        ),
    )

def _forward(self, x, emb):
    """
    Apply the block to a Tensor, conditioned on a timestep embedding.

    :param x: an [N x C x ...] Tensor of features.
    :param emb: an [N x emb_channels] Tensor of timestep embeddings.
    :return: an [N x C x ...] Tensor of outputs.
    """
    h = self.in_layers(x)
    emb_out = self.emb_layers(emb).type(h.dtype)
    while len(emb_out.shape) < len(h.shape):
        emb_out = emb_out[..., None]
    if self.use_scale_shift_norm:
        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        scale, shift = th.chunk(emb_out, 2, dim=1)
        h = out_norm(h) * (1 + scale) + shift
        h = out_rest(h)
    else:
        h = h + emb_out
        h = self.out_layers(h)
    return self.skip_connection(x) + h
```

![[adaptive-group-norm-annotated.excalidraw|1000]]