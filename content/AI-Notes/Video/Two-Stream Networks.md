# Two-Stream Networks

Tags: EECS 498, Model

The Two-Stream Network has two parallel convolutional neural network stacks.  

![[AI-Notes/Video/two-stream-networks-srcs/Screen_Shot.png]]

**Top Stack (Spatial Stream):**

Processes the appearance of the input video. This just takes a single frame from the input video and tries to predict a distribution of class scores.

**Bottom Stack (Temporal Stream):**

This will process the temporal information. You compute the optical flow between all adjacent frames, which will give you $T-1$ displacement fields. Then, you take the horizontal and vertical derivative for each of these displacement fields. This gives you $2 * (T - 1)$ gradients with height $H$ and width $W$. You stack all the gradients together and you get an input of $2 * (T-1) \times H \times W$. 

You then use an early fusion approach to fuse all the optical flow fields in the first layer. You can then use a normal 2D CNN for the rest of the network.

**Combination:**

The spatial and temporal streams both predict class distributions. You can just take an average of these two.