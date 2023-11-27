---
tags: [flashcards]
source: [[Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics, Alex Kendall et al., 2017.pdf]]
summary: weighs multiple loss functions by considering the homoscedastic uncertainty of each task
---

[[Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics, Alex Kendall et al., 2017.pdf]] introduced the concept.

They propose an approach to train multi-task deep learning models with an approach that weighs multiple loss functions by considering the [[Homoscedasticity|homoscedastic uncertainty]] of each task.

Prior approaches to simultaneously learning multiple tasks use a naive weighted sum of losses, where the loss weights are uniform, or manually tuned. Their model can learn multi-task weightings and outperform separate models trained individually on each task.

In Bayesian modelling, there are two main types of uncertainty one can model.
- **[[Aleatoric vs. Epistemic Uncertainty|Epistemic uncertainty]]** is uncertainty in the model, which captures what our model does not know due to lack of training data. It can be explained away with **increased training data**.
- **[[Aleatoric vs. Epistemic Uncertainty|Aleatoric uncertainty]]** captures our uncertainty with respect to information which our **data cannot explain**. Aleatoric uncertainty can be explained away with the ability to observe all explanatory variables with in- creasing precision.

**Aleatoric uncertainty** can again be divided into two sub-categories.
- Data-dependent or Heteroscedastic uncertainty is aleatoric uncertainty which depends on the input data and is predicted as a model output.
- **Task-dependent or Homoscedastic uncertainty** is aleatoric uncertainty which is not dependent on the input data. It is not a model output, rather it is a quantity which stays constant for all input data and varies between different tasks. It can therefore be described as task-dependent uncertainty.

> [!note] Homoscedastic uncertainty is where the variance between input and output varies between different tasks (ex. occupancy or velocity regression), but stays constant for all input examples.
> 
The paper learns an observation noise that they use in their loss functions.