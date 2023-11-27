---
tags: [flashcards]
source:
summary:
---

### For most layers
**Initializing the biases.** It is possible and common to initialize the biases to be zero, since the asymmetry breaking is provided by the small random numbers in the weights. For ReLU non-linearities, some people like to use small constant value such as 0.01 for all biases because this ensures that all ReLU units fire in the beginning and therefore obtain and propagate some gradient. However, it is not clear if this provides a consistent improvement (in fact some results seem to indicate that this performs worse) and it is more common to simply use 0 bias initialization. [Source]([http://cs231n.github.io/neural-networks-2/](http://cs231n.github.io/neural-networks-2/))

### For the final layer
Initialize the final layer weights correctly. E.g. if you are regressing some values that have a mean of 50 then initialize the final bias to 50. If you have an imbalanced dataset of a ratio 1:10 of positives:negatives, set the bias on your logits such that your network predicts probability of 0.1 at initialization. **Setting these correctly will speed up convergence and eliminate “hockey stick” loss curves where in the first few iteration your network is basically just learning the bias.**

**Why should you use a bias of 50 if the mean value of your regression is 50?**
Usually, in a classification network, the last layer is a softmax with `n` outputs (one for each class). By setting the appropriate bias we can make the model predict `1/n` for each class at initialization. This is because at initialization the layers have random values with a certain $\sigma$ and $\mu=0$. Therefore, after passing the input through the layers, the input to the last layer will have $\mu=0$ as well. The output of the final layer is

$$
\hat y_k = \text{softmax} (W z + b_k)
$$

where $z$ is the input, $W$ are the weights at initialization, and $b_k$ is the bias. Notice that $b_k$ is a vector of biases, one for each class. As we showed we expect $z$ to have $\mu = 0$. Therefore, we can control the output of the layer by tunning $b_k$. If we set 

$$b_k = b = 50$$

ie the same bias for each class, we will make the softmax output in average a probability of `1/n` for each input of the model.