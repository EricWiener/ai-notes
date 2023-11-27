---
tags: [flashcards, eecs498-dl4cv]
source:
summary: three viewpoints behind a linear classifier (algebraic, visual, and geometric).
---

![[three-viewpoints-linear-classifier.png]]

We will write a function $f$ that takes in $x$, the pixels of the image, and $W$ weights. 

$f(x, W)$ will give class scores.
- $f(x, W) = Wx + b$

Here are three different viewpoints of what this means:

## Cases a linear classifier can't learn
![[cases-a-linear-classifier-cant-learn.png]]