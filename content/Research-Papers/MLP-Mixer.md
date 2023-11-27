---
tags: [flashcards, eecs498-dl4cv]
source: [[MLP-Mixer_ An all-MLP Architecture for Vision, Ilya Tolstikhin et al., 2021.pdf]]
summary: Replaces the attention module with spatial MLPs as token mixers and finds the MLP-like model is competitive on image classification benchmarks.
---

MLP-Mixer is an all-MLP architecture for vision transformers. In a traditional [[Transformer]], you mix token patches with self-attention and then you apply a MLP independendly on each output from the self-attention layer.
![[transformer-w-attention.png]]
With MLP-Mixer, you mix across tokens with a MLP (instead of self-attention).
![[transformer-w-mlp.png]]
With MLP Mixer, you get $N$ input patches which are linearly embedded to have dimension $D$. This gives you $N \times D$ patch embeddings. 

For the token mixer you use a MLP that takes in $N$ channels and produces $N$ channels. You apply this independently to each of the $D$ input **channels** (operates on a single channel across all tokens to mix spatial information). This will give you an $N$ outputs each with dimension $D$.

You then apply another MLP on the $N$ tokens of dimension $D$. This MLP takes in $D$ channels and produces $D$ channels and mixes channel information. You will end up with $N \times D$ outputs where each output has been mixed first spatially and then along channels.

![[mlp-diagram.png]]
In the above diagram from the paper:
- You start out with $N$ patches (each color represents a different patch) of $C$ channels each represented in an $N \times C$ input.
- You transpose the $N \times C$ input to $C \times N$.
- You apply a MLP (C -> C) to each of the $N$ patches. This gives you a $C \times N$ output and mixes token information (operates on each channel independently).
- You transpose the $C \times N$ output back to $N \times C$.
- You apply a MLP (N -> N) to each of the $C$ channels. This gives you a $N \times C$ final output and mixes channel information (operates on each patch independently).

### Notes
![[mlp-mixer-is-weird-conv.png]]

- A MLP-Mixer is just a CNN. 
- The first MLP you use is a [[Pointwise Convolution|1x1 conv]] (takes C input channels, produces C output channels, and has a 1x1 receptive field).
- The second MLP is like a [[Grouped Convolutions]] with $C$ groups (each group operates independently on a channel). The receptive field is the entire input image (ex. $N^{1 / 2} \times N^{1 / 2}$ grid since $N^{1 / 2} \times N^{1 / 2} = N$ total patches). It shares the same filter across all the patches.

### Results
- Cool idea, but initial ImageNet results weren't great. Better results with JFT pretraining (similar to [[ViT An Image is Worth 16x16 Words Transformers for Image Recognition at Scale|ViT]]).