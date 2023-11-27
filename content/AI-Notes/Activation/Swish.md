---
tags: [flashcards]
source: [[Swish.pdf]]
summary: Slightly better than ReLU. $f(x)=x \cdot \sigma(x)$. Good for deep networks
---

Swish is just f(x) = x * sigmoid(x):

$$
f(x)=x \cdot \sigma(x)
$$
![[swish-graph.png.png]]

### Paper Highlights
We believe that the properties of Swish being unbounded above, bounded below, non-monotonic, and smooth are all advantageous.

### Additional Notes
Note: if using int8 quantization, you will need to truncate your activation values to preserve negative values. Nvidia has a white paper on this.