# Spatio-Temporal Self-Attention

Tags: EECS 498, Layer

We can use attention to merge spatial and temporal information. 

### Attention Review:
![[AI-Notes/Video/spatio-temporal-self-attention-srcs/Screen_Shot_1.png]]

Previously when we used self-attention for a 2D CNN, we had the following diagram:

![[Self-Attention for 2D CNN]]

Self-Attention for 2D CNN

## 3D Self-Attention

Now, we will need to replace the 2D Conv with a 3D Conv.

![[AI-Notes/Video/spatio-temporal-self-attention-srcs/Screen_Shot 1.png]]

We now have a block that we can insert into our 3D CNNs that computes a temporal-spatial self-attention. This is sometimes called a **Nonlocal Block** (after the paper that it was published in).

**Trick:** in practice, what is often done is to insert this block into a 3D CNN with the conv weights shown in blue initialized to all zeros. Because of the residual connection, setting these weights to all zeros causes this block to just be the identity. You can then contain training and fine-tune using the additional block.

When we insert the block in between 3D convolutions, we the 3D convolutions will perform a slow fusion over space and time. Each nonlocal block will give a global fusion over both space and time.