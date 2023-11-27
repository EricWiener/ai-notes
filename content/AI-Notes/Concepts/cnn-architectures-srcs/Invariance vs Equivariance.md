---
tags: [flashcards]
aliases: [invariance, equivariance, permutation invariant, permutation equivariant]
---
![[invariance-vs-equivariance-20220223145332004.png]]

==Invariance== means that the output is not affected by translational changes in the input. Here, the cat is translated, but the output label is still “cat”. ==Equivariance==, on the other hand, is shown in this image on the right. The input image is translated and the output varies “equivalently” in that the semantic segmentation of cat vs background is shifted by the same amount as the input. By design of convolutional kernels, CNNs contain both of these benefits.
<!--SR:!2027-07-02,1511,350!2027-06-21,1502,350-->

It is usually trivial to convert an equivariant response into an invariant one--just disregard all the position information