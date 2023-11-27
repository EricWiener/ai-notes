# Model Ensemble

Files: 498_FA2019_lecture11.pdf

# Model Ensemble

If you want to get slightly better performance on most problems, you can train $n$ independent models. At test time you can average their results.

You usually get about 1-2% percent boost on the test set.

Instead of training multiple independent models, you can also use multiple checkpoints of your model while it was training. The training for this type of setup is sometimes done with a cyclic learning rate schedule. You choose to use the checkpoints at each of the low points of the cyclic function.

![[cyclic-learning-rate.png]]


You can also use [[Checkpoint Averaging|checkpoint averaging]] to average model weights to smooth out variance.