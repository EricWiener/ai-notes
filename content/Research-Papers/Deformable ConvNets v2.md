---
tags: [flashcards]
source: https://arxiv.org/abs/1811.11168
aliases: [Deformable ConvNets v2: More Deformable, Better Results]
summary: introduces [[Modulated Deformable Convolution|Deformable Convolution v2]]
---

[[Deformable Convolution]] conforms more closely than reg- ular ConvNets to object structure, this support may never- theless extend well beyond the region of interest, causing features to be influenced by irrelevant image content. This paper introduces [[Modulated Deformable Convolution|Deformable Convolution v2]] which improves its ability to focus on important image regions, through increased modeling power and stronger training.

The paper uses [[Distillation]] from R-CNN which takes region proposals as input so the features it extracts only depend on the relevant region. By matching the features extracted from R-CNN, the deformable convolutions will also learn to focus on relevant information.

The paper improves on Deformable ConvNets (the paper that introduced [[Deformable Convolution]]) by increasing the number of deformable layers in the network and adding a modulation mechanism to the deformable convolution modules where each sample undergoes a learned offset and is scaled by a learned feature amplitude.