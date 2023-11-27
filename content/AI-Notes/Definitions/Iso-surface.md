---
tags: [flashcards]
source:
summary: a surface defined by all points that take the same value for a function of space
---

Isosurface is another way to call a surface defined by the implicit equation
$$F(x, y, z)=f$$
where $F$ is a function of space and $f$ a constant, often $0$. The prefix iso- indicates that the function $F$ takes the same value ($f$) all over the surface.

An example usage is if you have a neural network that takes an $(x, y, z)$ co-ordinate and predicts whether that cell is the border of an object. You can then query the network at an interval to reconstruct a mesh of a 3D scene.

![[iso-surface-20230619122830092.png]]