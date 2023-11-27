---
tags: [flashcards]
aliases: [Learning Lane Graph Representations for Motion Forecasting]
source: https://arxiv.org/abs/2007.13732
summary: this paper was referenced by [[Scene Transformer]] as using an agent-centric representation in a global frame
---

### ActorNet: Extracting Traffic Participant Representations
We assume actor data is composed of the observed past trajectories of all actors in the scene. Each trajectory is represented as a sequence of displacements $\left\{\Delta \mathbf{p}_{-(T-1)}, \ldots, \Delta \mathbf{p}_{-1}, \Delta \mathbf{p}_0\right\}$, where $\Delta \mathbf{p}_t$ is the 2D displacement from time step $t-1$ to $t$, and $T$ is the trajectory size. All coordinates are defined in the Bird's Eye View (BEV), as this is the space of interest for traffic agents.