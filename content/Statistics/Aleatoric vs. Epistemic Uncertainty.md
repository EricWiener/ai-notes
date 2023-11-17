---
tags:
  - flashcards
source: https://ar5iv.labs.arxiv.org/html/2111.14973
summary: 
aliases:
  - aleatoric uncertainty
  - epistemic uncertainty
---
### Aleatoric uncertainty
**Aleatoric uncertainty**: This is a ==natural variation== in the data. For example an agent can either take a left or right turn or change lanes, etc given the same context information.

This level of ambiguity cannot be resolved by increasing the model capacity, but rather the model needs to predict calibrated probabilities for these outcomes. Despite the theoretical possibility of modeling these variations using a small number of output trajectories directly, there are several challenges in learning. Some examples include mode collapse and failure to model these variations due to limited model capacity.

See [[Homoscedastic loss]] for more info.

### Epistemic uncertainty
**Epistemic uncertainty**: This is the variation across ==model outputs==, which typically indicates the model’s failure to capture certain aspects of the scene or input features. 

Such variations could occur if some models are poorly trained or haven’t seen a particular slice of the data. By doing model ensembling, we attempt to reduce this uncertainty.

