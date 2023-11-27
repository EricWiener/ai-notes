---
tags: [flashcards]
source: https://youtu.be/aw3H-wPuRcw
summary: a quick summary of attention and cross-attention
---

Attention avoids uses hard-coded rules and human intervention and lets the model decide how to incorporate global information.

The formula for attention or self-attention is:
$$\operatorname{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d}}\right) \cdot V$$
You take an input and construct three separate matrices for Queries, Keys, and Values. You divide by a scalar $\sqrt{d}$ to stabilize training.

### Attention vs. Convolution
![[attention-vs-convolution-receptive-field.png]]
Attention has a global receptive field and can attend to all pixels in an image (or in a patch of an image) while a convolution can only see the receptive field around it.

# Attention Computation
### Project input to $Q, K, V$
You first flatten your 2D image into a 1D array. You then project your flattened sequence of shape `[num_pixels, num_channels]` using the weights $W^Q, W^K, W^V$. This will give you your queries, keys, and values. 
![[Cross Attention _ Method Explanation _ Math Explained 3-12 screenshot.png]]
In the above example, we have a 2x2 RGB image that we flattened into a (4, 3) matrix (4 pixels and 3 color channels). We then apply the weight matrices with shape (3, 2) which will project our input from having 3 channels to have 2 channels (typically the number of channels we project to is much greater).

### Calculating $Q \cdot K^T$
This will compute the dot product between each row of $Q$ and each column of $K$ which will compute the dot product between each pair of pixels (ex. the first row of $Q$, `[0.28, 0.12]` is one pixel query and the dot product will be taken with `[0.07, 0.21]` which is the same pixel's key values). Each value in the resulting matrix gives you a single matrix indicating the similarity between each pair of pixels key and value embedding. This gives us a **similarity matrix of each pixel to every other pixel**.
![[screenshot-2023-06-18_11-01-52.png|500]]

A visual interpretation of what this is doing is below. $Q$ and $K$ project each pixel into an embedding. We then take the dot product between each pair of pixels query and key value. If the two embeddings are close to each other in the embedding space, then they will have a larger dot product. If they are further away, then they will have a smaller dot product. This lets the model learn via the weight matrices $Q$ and $K$ how close in the embedding space it wants two pixels to be and therefore how large a resulting weight to give.
![[attention-quick-summary-20230618110631143.png|600]]

### Applying softmax to $\frac{Q \cdot K^T}{\sqrt{d}}$
You then apply a softmax operation on each row of the resulting vector. Each row is the dot product between one pixel and all other pixels. Taking the softmax normalizes the dot product values into weights that sum to 1. This tells us how much one pixel will attend to all other pixels. For instance, the first row of the resulting matrix tells us the 0th pixel will attend 23% to itself, 33% to the second pixel, 27% to the third pixel, and 17% to the fourth pixel.
![[apply-softmax-to-similarity-matrix.png|500]]

This shows how attention lets us route information between all pixels in a single layer.

### Multiply similarity matrix by $V$ matrix
This basically acts as a weighted average between the weights we computed via $\operatorname{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d}}\right)$ and the values $V$.
![[screenshot-2023-06-18_11-19-15.png|500]]
To compute the top-left element in the resulting matrix, you do:
$$(0.23 * -1.03) + (0.33 * -0.67) + (0.27 * -0.56) + (0.17 * -1.04) = -0.79$$
and then to compute the top-right element you do:
$$(0.23 * -0.62) + (0.33 * -1.46) + (0.27 * -0.97) + (0.17 * -1.04) = 0.9$$ 
Another way to interpret this is multiplying each row in the similarity matrix and multiplying each value by the corresponding row embedding for the corresponding pixel embedding in $V$.
![[screenshot-2023-06-18_11-20-46.png|500]]
This alternate way of computing the resulting matrices makes it clearer that you are computing a weighted sum over each pixel's value embedding in $V$ using a single pixels similarity weights (each row in $\operatorname{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d}}\right)$). 

Each row is the weighted average of the embeddings of all pixels where the proportions were determined by the model weights for the projection matrices $Q$ and $K$ which determined how close the embeddings where and therefore how large the dot product results were.

This is like being a chef and determining the proportions to combine the ingredients in a recipe to get the best meal.

### Project into original space
You then project the resulting matrix we get above back into the original inputs dimension using $W^{\text{out}}$. Here, $W^{\text{out}}$ has shape (2, 3) and will give us an output matrix with the same dimensions as our input.
![[screenshot-2023-06-18_11-25-04.png|500]]

### Add the attention output to the input
You then do $X + \operatorname{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d}}\right) \cdot V$. The reason for the skip connection is so the gradients can flow better but also so the attention can fully attend to the pixels it wants to without forcing each pixel to attend a lot to itself to retain information for later layers.

# [[Cross-attention]]
Cross Attention differs from regular attention in that only the queries, $Q$, come from input and the keys, $K$, and values, $V$, come from the conditional information.

In the following example, we will condition on text. Most models that condition on text will use a pre-trained transformer encoder. You take a caption, tokenize, add positional encodings and embed the tokens, and then pass it to the transformer encoder which gives you your encoded caption.

For instance, we can take the caption "I love mountains" and get the transformer encoder output for it. If the transformer encoder output dimension is 6, then we will get a matrix of shape (3, 6) where each word has a corresponding embedding of dimension 6.

![[attention-quick-summary-20230618123444702.png]]

We still have a 2x2 RGB image represented by a (4, 3) matrix. We also still have a query embedding matrix, $W^Q$, with shape (4, 3) to take us from the 3 RGB channels to the embedded dimension 2. Applying $W^Q$ to our input image, we get a resulting matrix of shape (4 pixels, 2).

For the text we want to condition on ("I love mountains"), we need to project from the 6 dimensions from our transformer encoder output to the same dimension as the embedded image, 2. Therefore, we have $W^K, W^V$ with shape (6, 2). We end up with a resulting matrix of shape (3 words, 2) for the keys and values.

> [!NOTE] You can condition on any type of information as long as you can project the information you want to condition on into a dimension of a reasonable size (to avoid memory and computation blowing up).

### Multiplying $K$ and $V$
![[screenshot-2023-06-18_12-35-45.png|400]]
You then compute the dot product between each row in $K$ (a single pixel's embedding) by each column in $V$ (a single token's embedding). The resulting matrix will have each pixel dot-producted with each word. You then normalize by dividing by $\sqrt{d}$ and applying a softmax to make each row add up to 1.

### Multiply similarity matrix by $V$ matrix
![[screenshot-2023-06-18_12-38-42.png|500]]

You then compute a weighted sum of each row of the similarity matrix (each row is a pixel and each element in the row is how much weight to give to a token) applied to the token value embeddings (each row in $V$ is a single token's embedding). The resulting matrix will contain information of the text conditioning where each pixel has chosen how much weight it wants to apply to each of the tokens.

### Project back to original input dimension
![[screenshot-2023-06-18_12-42-20.png|500]]

You then project back into the original image's dimension (3 channels) using $W^{\text{out}}$ with shape (2, 3).

### Residual connection
You can then add the resulting (4,3) matrix back to the original image of shape (4, 3). The result will contain information from both the original input and also information from the conditioned text where each pixel has decided how much weight to give to each token.