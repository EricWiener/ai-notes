---
tags: [flashcards]
source: [[StarNet_ Joint Action-Space Prediction with Star Graphs and Implicit Global Frame Self-Attention, Faris Janjoš et al., 2021.pdf]]
summary:
---

- Uses a graph view of the scene instead of a top-down scene (BEV)
- Trend in literature moves away from single agent prediction and towards multiple agents.

### VectorNet
![[screenshot-2022-03-03_19-09-00.png]]

- Similar paper from Waymo in 2020 (VectorNet). Scene is a big collection of vectors.
- Collect all vectors that form a feature (ex. lane line) into a polyline subgraph.
    - Get one feature vector per poly-line.
- Have a global interaction graph

# StarNet
- Avoids global interaction graph
- Try to use relative frames for multiple agents at once.
    - VectorNet tried to make frames relative to the hero (if I understood correctly)
- They do smaller amounts of computation in the reference frame of each vehicle and then try to aggregate

### Single Agent

![[screenshot-2022-03-03_19-10-47.png]]
- Superscripts mean they are in the reference frame of that agent
- Message passing goes through the central node (the agent - shown in blue)
- The yellow is the polyline.
- The paper doesn't think that the road features interacting together is important
- They actually care about the vectors of the polyline and the agent.
- The graphs are very agent centric.
- Each of the agent-centered graph goes through graph + attention network
- Multi-head attention portion:
    - $z^1_1$ is the agent itself w respect to itself
    - $q^1_2$ is the second road feature with respect to 1st agent.
- Take output embedding and pass it through a [[GRU]].
- Then pass to a kinematics model (basically an integrator) to avoid the model needing to learn the kinematics
- Trying to be more efficient with the graph representation
- Here you use the actual polyline vectors in the graph (vs. Waymo that used aggregated features).

### Multiple Agents
![[screenshot-2022-03-03_19-24-42.png]]
- The bottom attention is $O(n^4)$ in terms of $n$ agents
- They use a form of masked attention.
- Why did they compute polyline embeddings if they weren't used? They are used - they are attended to, but not passed on to later layers.

# Results
- They use the Interaction dataset
- This approach is more agent-centric, less global, and do more with the road features.

# Questions
- [ ] What is a polyline?
- [ ] What is a GAT?
- [ ] What is a GRU?