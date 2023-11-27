---
source: https://paperswithcode.com/method/causal-convolution
tldr: convolution used for temporal data to ensure future predictions only depend on previous data.
tags: [flashcards]
---

**Causal convolutions** are a type of [convolution](https://paperswithcode.com/method/convolution) used for ==temporal data== which ensures the model cannot ==violate the ordering in which we model the data==: the prediction $p\left(x_{t+1} \mid x_{1}, \ldots, x_{t}\right)$ emitted by the model at timestep t cannot depend on any of the future timesteps $x_{t+1}, x_{t+2}, \ldots, x_{T}$. For images, the equivalent of a causal convolution is a [[Masked Convolution]] which can be implemented by constructing a mask tensor and doing an element-wise multiplication of this mask with the convolution kernel before applying it. For 1-D data such as audio one can more easily implement this by shifting the output of a normal convolution by a few timesteps.
<!--SR:!2024-12-04,657,270!2024-07-03,456,308-->

![[casual-convolutions-20220227091830271.png]]

