---
tags: [flashcards]
aliases: [PQ]
source: https://arxiv.org/abs/1801.00868
summary: measures the quality of a predicted panoptic segmentation relative to the ground truth
---

**Main steps**:
1. Segment matching
2. PQ computation given the matches.

### Segment Matching
- A predicted segment and a ground truth segment can match only if their intersection over union ([[IoU]]) is greater than 0.5.
- Since panoptic segmentation produces nn-overlapping masks, this requirement makes it so that there can be at most one predicted segment matched with each ground truth segment.
- Due to the uniqueness property, for IoU > 0.5, any reasonable matching strategy (including greedy and optimal) will yield an identical matching. Lower thresholds are unnecessary as matches with IoU ≤ 0.5 are rare in practice, so greedy matching can be used.

### PQ Computation
- You calculate PQ for each class independently and average over the classes. This makes PQ insensitive to class imbalance.

$$
\mathrm{PQ}=\frac{\sum_{(p, g) \in T P} \operatorname{IoU}(p, g)}{|T P|+\frac{1}{2}|F P|+\frac{1}{2}|F N|}
$$
- $\frac{1}{|T P|} \sum_{(p, g) \in T P} \operatorname{IoU}(p, g)$ is the average IoU of matched segments.
- $\frac{1}{2}|F P|+\frac{1}{2}|F N|$ is added to the denominator to penalize segments without matches.
- All segments receive equal importance regardless of their area.

### PQ Interpretation
![[pq-sq-dq.png]]
If we multiply and divide PQ by the size of the TP set, then PQ can be seen as the multiplication of a **Segmentation Quality** (SQ) term and a **Detection Quality** (DQ) term.

The SQ term measures how good the segmentation mask is and the detection quality measures if the model misses any instances (FN) or predicts additional instances (FP).

### Handling void labels
Void labels come from:
- Out of class pixels (classes out of the domain you care about).
- Ambigious or unknown pixels.

Void pixels are handled specially during matching and computation:
> Specifically: (1) during matching, all pixels in a predicted segment that are labeled as void in the ground truth are removed from the prediction and do not affect IoU computation, and (2) after matching, unmatched predicted segments that contain a fraction of void pixels over the matching threshold are removed and do not count as false positives.

