---
tags: [flashcards]
source: [[Deep Roots- Improving CNN Efficiency with Hierarchical Filter Groups.pdf]]
aliases: [Deep Roots, Filter Groups]
summary: Use filter groups (divide up channel dimension) for similar/higher accuracy than the baseline architectures with much less computation.
---

### Standard [[Convolutional Neural Networks|CONV layer]]

![[multiple-convolutional-layers.png]]

Usually in a CONV layer, the filter will have the same number of channels as the input and the number of filters you use determines the output number of channels. This means each filter operates on all  input channels.

### Filter Groups
![[filter-groups.png]] 
Figure 1: **Filter Groups.** (a) Convolutional filters (yellow) typically have the same channel dimension c1 as the input feature maps (gray) on which they operate. However, (b) with filter grouping, g independent groups of $c_2/g$ filters operate on a fraction $c_1/g$ of the input feature map channels, reducing filter dimensions from $h \times w \times c_{1}$ to $h \times w \times c_{1} / g$.

> [!note]
> Each filter now operates on a fraction of the input channels. You end up with the same input and output number of channels, but each filter can be $1/g$ of the original size.