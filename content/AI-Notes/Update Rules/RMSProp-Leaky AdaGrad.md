# RMSProp: Leaky AdaGrad

Files: lecture_slides_lec6.pdf
url: https://cs231n.github.io/neural-networks-3/#sgd

RMSProp adjusts the Adagrad method to reduce its aggressive monotonically decreasing learning rate, but allowing the sum of squares of the gradients to decrease.

It is currently unpublished and everyone who uses this method in their work currently cites [slide 29 of Lecture 6](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) of Geoff Hinton’s Coursera class.

![[RMSProp Le/Screen_Shot.png]]

You add a decay rate that shrinks the sum of the gradients. This prevents us from slowing down too much.

We square the grad because it has nice properties for convex optimizations (this is why we don't just use the absolute value of the gradient).

```python
# Update the cache, but also decay both the original
# cache and the squared gradient values you add
cache = decay_rate * cache + (1 - decay_rate) * dw**2

# Update w the same as you did with Adagrad
w -= learning_rate * dw / (np.sqrt(cache) + eps)
```