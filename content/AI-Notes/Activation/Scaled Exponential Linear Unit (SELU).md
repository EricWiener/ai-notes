---
summary: Scaled version of ELU that works better for deep networks (self-normalizing property). Allows training without needing BatchNorm
---

Scaled version of ELU that works better for deep networks. It has a "self-normalizing" property that can train deep SELU networks without needing BatchNorm because as the number of layers goes to infinity, the statistics of your activations will be well behaved and converge to a finite value.

![[scaled-exp-math.png.png]]

The derivation takes 91 pages of math in the appendix.