---
tags: [flashcards]
source: https://www.youtube.com/watch?v=zCEYiCxrL_0
summary: practical tips to debug ML model
---

Model Capacity (what can the model learn?)
- Overtrain on a small dataset
- Synthetic data

Optimization Issues (can we make the model learn?)
- Look at learning curves
- Monitor gradient update ratios
- Hand-pick parameters for synthetic data
- Other model "bugs" (is the model doing what I want it to do?)
- Generate samples from your model (if you can)
- Visualize learned representations (e.g. embeddings, nearest neighbors)
- Error analysis (examples where the model is failing, most "confident" errors)
- Simplify the problem/model
- Increase capacity, sweep hyperparameters (e.g. increase size of h in LSTM)