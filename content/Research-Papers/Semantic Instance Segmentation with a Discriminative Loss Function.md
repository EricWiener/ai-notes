---
tags: [flashcards]
source: https://arxiv.org/abs/1708.02551
summary: paper for semantic segmentation that predicts a per-pixel instance embedding to allow clustering instances
---

Uses a discriminative loss to calculate pixel level embeddings.

[Learning a Spatio-Temporal Embedding for Video Instance Segmentation](https://arxiv.org/abs/1912.08969) is an extension for video instance segmentation.

![[semantic-instance-segmentation-with-a-discriminative-loss-function-20230507194708461.png]]

![[semantic-instance-segmentation-with-a-discriminative-loss-function-20230507194714537.png]]
Figure 1:The network maps each pixel to a point in feature space so that pixels belonging to the same instance are close to each other, and can easily be clustered with a fast post-processing step. From top to bottom, left to right: input image, output of the network, pixel embeddings in 2-dimensional feature space, clustered image.

### Discriminative loss function
![[semantic-instance-segmentation-with-a-discriminative-loss-function-20230507194743352.png]]

Note: this seems very similar to [[Contrastive Learning|Contrastive Loss]].

Weinberger _et al_. [[39](https://ar5iv.labs.arxiv.org/html/1708.02551#bib.bib39)] propose a loss function with two competing terms to achieve this objective: a term to penalize large distances between embeddings with the same label, and a term to penalize small distances between embeddings with a different label.

In our loss function we keep the first term, but replace the second term with a more tractable one: instead of directly penalizing small distances between every pair of differently-labeled embeddings, we only penalize small distances between the mean embeddings of different labels. If the number of different labels is smaller than the number of inputs, this is computationally much cheaper than calculating the distances between every pair of embeddings. This is a valid assumption for instance segmentation, where there are orders of magnitude fewer instances than pixels in an image.

We now formulate our discriminative loss in terms of push (i.e. repelling) and pull forces between and within clusters. A cluster is defined as a group of pixel embeddings sharing the same label, e.g. pixels belonging to the same instance. Our loss consists of three terms: