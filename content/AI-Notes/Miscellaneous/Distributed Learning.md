# Distributed Learning

Files: 498_FA2019_lecture11.pdf

This type of learning makes use of multiple GPUs to speed up training a model. This is called **model parallelism.**

**Idea 1: Run different layers on different GPUs**

![[AI Notes/Miscellaneous/Distribute/Screen_Shot.png]]

This idea doesn't work very well because the GPUs end up waiting for each other.

**Idea 2: Run parallel branches of model on different GPUs**

![[AI Notes/Miscellaneous/Distribute/Screen_Shot 1.png]]

This was used for the original AlexNet paper. It requires a lot of synchronizing between GPUs, which is expensive. It needs to communicate the activations and gradients with respective to those activations during the forward and backward pass.

You are passing the whole minibatch through both GPUs, but areas of the model that are parallel, can be run separately (ex. parallel residual blocks).

**Idea 3: Use data parallelism to train multiple full models on split up minibatches**

![[AI Notes/Miscellaneous/Distribute/Screen_Shot 2.png]]

You split up your minibatch of $N$ images over as many GPUs as you have. The computation can now be fully done in parallel. You only need to communicate once per iteration when you sum the gradient of the loss with respect to all the different parameters in order to make a gradient step.

> [!note]
> This is how people typically make use of multiple GPUs
> 

You can now train for the same amount of time, but use larger minibatches. You can usually scale the batch size and learning rate with the number of GPUs. 

- If on a single GPU you use use batch size $N$ and learning rate $\alpha$
- On a $K$ GPU model, you use batch size $KN$ and learning rate $K\alpha$
- This very large learning rate can often cause the loss to explode in the first couple iterations of training. People often use a learning rate schedule that has a linearly **increasing** learning rate from 0 over the first ~5000 iterations.