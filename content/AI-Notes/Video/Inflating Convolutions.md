---
tags: [flashcards, eecs442, eecs498-dl4cv]
source:
summary:
---
There has been a lot of work on creating the best 2D architectures. We want to take these architectures and adapt them to make them work on videos.

### Inflated Convolutions: ResNet

![[AI-Notes/Video/inflating-convolutions-srcs/Screen_Shot.png]]

![[AI-Notes/Video/inflating-convolutions-srcs/Screen_Shot 1.png]]

You can start with a 2D architecture like ResNet and just replace the $3 \times 3$ filters with $3 \times 3 \times 3$. You can even reuse the weights and pre-train with 2D models.

### Reusing weights

![[AI-Notes/Video/inflating-convolutions-srcs/Screen_Shot 2.png]]

In order to copy the weights from the pre-trained 2D model to the 3D model, we can just copy the weights for each temporal channel. You need to copy the weights $K_t$ times for each temporal channel. Then, you divide by $K_t$.

**Why divide by** $K_t$:

If you imagine you took an input image and then just repeated it a lot of times, you would get a very boring movie. If you convolved a 2D filter with it, you would end up with the same results as the 3D filter. Dividing by $K_t$ ensures this happens. Convolution is a linear operation, so if you add up $K_t$ weights, you then need to divide by $K_t$. You want to get the same result because this is how you are able to re-use weights. If you got a different result, the weights wouldn't be re-usable.

### Filter Seperation

However, these filters are now pretty big. We want to reduce the size of the filters, so we can separate the filters.

![[AI-Notes/Video/inflating-convolutions-srcs/Screen_Shot 3.png]]

One filter will focus on time $3 \times 1 \times 1$, while the other filter will focus on space $1 \times 3 \times 3$. This makes sense since time and space likely have different features. This often works well and is faster + has fewer parameters.