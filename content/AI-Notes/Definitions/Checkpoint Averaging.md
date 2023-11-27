---
tags: [flashcards]
source: 
summary: averages the model weights from a couple of checkpoints
---

Checkpoint Averaging averages the ==model weights== from (usually) the last couple training checkpoints. This can help ==smooth out some of the variance== from iteration to iteration.
<!--SR:!2024-02-25,559,330!2028-01-26,1643,330-->

[Fairseq Implementation](https://github.com/pytorch/fairseq/blob/main/scripts/average_checkpoints.py)

Tensorflow description: The advantage of Moving Averaging is that they are less prone to rampant loss shifts or irregular data representation in the latest batch. It gives a smooothened and a more genral idea of the model training until some point. [Source](https://www.tensorflow.org/addons/tutorials/average_optimizers_callback)

Relevant paper: [[Checkpoint Ensembles_ Ensemble Methods from a Single Training Process, Hugh Chen et al., 2017.pdf]]