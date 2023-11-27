```python
import numpy as np
import seaborn as sns

def sinusoid_positional_encoding_ref(length, dimensions):
  def get_position_angle_vec(position):
    return [position / np.power(10000, (i - 1 if i % 2 == 1 else i) / dimensions) for i in range(dimensions)]

  PE = np.array([get_position_angle_vec(i) for i in range(length)])
  PE[:, 0::2] = np.sin(PE[:, 0::2]) # dim 2i
  PE[:, 1::2] = np.cos(PE[:, 1::2]) # dim 2i+1
  return PE

pe = sinusoid_positional_encoding_ref(200, 12)

sns.set(rc={'figure.figsize':(14,4)})
ax = sns.heatmap(pe.T)
ax.invert_yaxis()
ax.set_ylabel("dimension in token (i)")
ax.set_xlabel("token index (pos)")
ax.set_title("Sinusoid absolute positional encoding")
```