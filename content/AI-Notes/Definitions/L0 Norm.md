---
tags: [flashcards]
source: https://montjoile.medium.com/l0-norm-l1-norm-l2-norm-l-infinity-norm-7a7d18a4f40c
summary: the total number of nonzero elements in a vector.
---
The L0 norm is the total number of ==nonzero elements== in a vector.
<!--SR:!2024-05-13,596,330-->

For example, the L0 norm of the vectors (0,0) and (0,2) is 1 because there is only one nonzero element.

The L0 norm is useful if dealing with features that aren't standardized. L1/L2 norm will shrink weights to be small. However, you might want to just decrease some values to 0, but not get rid of having large values. L0 will shrink some values to 0, but will still allow you to have large values.