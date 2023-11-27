---
tags: [flashcards]
aliases: [Occupancy Flow Fields, OccFlow, Backwards Flow]
source: https://arxiv.org/abs/2203.03875
summary: this paper introduces predicting backward flow which allows representing multiple next destinations for any current occupancy.
---

Flow predictions predict the movement of an agent in a grid cell at a particular timestep. They are useful because instead of needing to predict many discrete occupancy "key frames," you can instead predict a couple of flow fields and then interpolate between key frames to find where an agent is at any continuous point in time.

Using backward flow fields have multiple benefits over forward flow fields. They allow representing multiple next destinations for an agent, predicting where initially occluded agents appear from, and having meaningful flow predictions for all grid cells vs. just the grid cells that are predicted to contain an occupied agent.

# Backward flow overview
![[occupancy-flow-fields-for-motion-forecasting-in-autonomous-driving-20230619124327792.png]]
The above diagram shows an example forward flow field on the left and a backward flow field on the right. With the forward flow, each grid cell predicts its future location. With the backward flow each grid cell predicts its past location. A single backward flow field can represent multiple next destinations for any current occupancy which makes it more effective for motion forecasting since it can capture multiple futures for individual agents using a ==single flow field== per timestep. Predicting multiple futures with forward flows requires predicting multiple flow vectors per cell and their associated probabilities which increases latency, memory requirements, and model complexity.
<!--SR:!2024-12-19,418,310-->

Backward flow fields are meaningful and predicted on every currently-unoccupied cell in the map (not shown), but forward flow is only meaningful on current occupancies.

Additionally, each backward flow vector can be traced back to a single $t = 0$ occupancy frame which means that each flow field will trace back to a ==single agent== unlike forward flow fields which could end up pointing to an overlapping region and predicting agents colliding with each other (see the diagram below with 2 timesteps).
![[forward-vs-backward-flow.excalidraw]]
Tracing the flow fields backwards allows recovering the identify of the agent predicted to occupy any future or current grid cell.
<!--SR:!2024-12-22,421,310-->

> [!NOTE] Backward flow still models the forward motion of agent
> Both backward and forward flow model the forward motion of agents. However, backward flow represents where each grid cell comes from in the previous timestep rather than representing where each grid cell moves to in the next timestep.

### Math and notation
Occupancy flow fields can be represented as two quantities: an occupancy grid $O_t$ and a flow field $F_t$, both with spatial dimensions $h \times w$. Each cell in the grid corresponds to a particular BEV grid cell in the map.
- Each cell $(x, y)$ in $O_t$ contains a value in $[0, 1]$.
- Each cell $(x, y)$ in $F_t$ contains a two dimensional vector $(\Delta x, \Delta y)$ that specifies the motion of any agent whose box occupies the grid cell at time $t$. The units of the flow vectors are in grid cell units.

Vehicle and pedestrian behavior are very different, so occupancy and flow predictions are done separately for the $K$ different agent classes. The paper predicts occupancy grids $O_t=\left(O_t^V, O_t^P\right)$ and flow fields $F_t=\left(F_t^V, F_t^P\right)$ for vehicles and pedestrians. $\forall t \in \{1, \ldots, T_\text{pred}\}$.

**Ground truth**
The ground truth flow vectors between times $t$ and $t-1$ are placed in the grid at time $t$ and point to the original position of that grid cell at time $t - 1$.

This can be calculated as $\tilde{F}_t(x, y)=(x, y)_{t-1}-(x, y)_t$ where $(x, y)_{t-1}$ is the coordinates at time $t - 1$ of the agent that occupies $(x, y)$ at $t$.

# Occluded Agents
Backward flow is also able to handle occluded agents that have not been observed in the past since you can predict where an agent came from. This is useful for planning purposes since you can reason where an agent was in an earlier frame. This isn't possible with forward flow since you don't have an initial occupancy prediction in an earlier frame.

The paper trains the model with alternative labels that reflect occupancy and flow of agents known to exist in the future but missing in past timesteps.

