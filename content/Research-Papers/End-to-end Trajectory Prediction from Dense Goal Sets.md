---
tags: [flashcards]
source: https://arxiv.org/abs/2108.09640
aliases: [DenseTNT]
summary:
---

DenseTNT improves on existing approaches for [[Goal Based Trajectory Prediction]]. It uses a goal-based strategy to classify the endpoint of a trajectory from a set of dense goal points (source: [[Motion Transformer with Global Intention Localization and Local Movement Refinement|MTR]]).

**How is DenseTNT different from existing goal-based trajectory prediction methods?**
??
DenseTNT doesn't rely on anchors (like selecting goals from points along a lane centerline). It instead predicts goal confidences for a dense set of possible goals. It uses a model to narrow down the predicted goals to the final set of goals to generate trajectories for (vs. using a heuristic like NSM).
![[densetnt-trajectory-prediction.png|500]]
<!--SR:!2024-03-05,156,250-->

# Existing Work
Existing approaches use heuristics for anchors for possible goal locations (ex. TNT defines anchors as the points sampled on lane centerlines and others use lane segments as anchors). They also commonly use a rule-based algorithm to select a final small number of goals like using [[Non-maximum Suppression]] to select only high-scored goals.

**Downsides:**
These approaches have two main downsides:
1. The prediction performance relies heavily on the quality of the goal anchors.
2. The heuristics used to narrow down the final selected goals are not guaranteed to find the optimal solution given the multi-modal nature of prediction.

# Approach
DenseTNT is anchor-free and predicts multiple trajectories end-to-end. It first generates dense goal candidates with their probabilities from the scene context and then it uses a goal set predictor to produce a final set of trajectory goals.

In motion prediction, unlike perception, you only observe one ground truth future out of multiple possible futures in each training sample (in perception there is only one correct solution). This makes it challenging to train the model. To solve this problem, they use an offline model to provide multi-future pseudo-labels for the online model. The offline model uses an optimization algorithm instead of the goal set predictor for goal set prediction. The optimization algorithm finds an optimal goal set from the probability distribution of the goals; and then the set of goals are used as pseudo-labels for the training of the online model.