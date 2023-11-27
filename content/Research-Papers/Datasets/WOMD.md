---
tags:
  - flashcards
aliases:
  - Waymo Open Motion Dataset
source: https://waymo.com/open/data/motion/
summary: a large motion prediction dataset where you need to predict independent tracks for up to 8 agents surrounding hero / scene prediction for 2 agents per scene.
publish: true
---

The WOMD dataset consists of 104,000 run segments (each 20 seconds in length) from 1,750 km of roadway containing 7.64 million unique agent tracks. Importantly, the WOMD has two tasks, each with their own set of evaluation metrics: a marginal motion prediction challenge that evaluates the quality of the motion predictions independently for each agent (up to 8 per scene), and a joint motion prediction challenge that evaluates the quality of a model’s joint predictions of exactly 2 agents per scene. (Source: [[Scene Transformer]])