---
tags: [flashcards]
source:
aliases: [MTR]
summary:
---

Motion forecasting requires predicting the future behavior of traffic participants by jointly considering the observed agent states and the road maps. It is challenging since agents have multimodal behavior (ex. an agent at an intersection can turn left, turn right, or proceed) and scenes can be complex.

This paper takes the approach of first predicting a general target for an agent and then refining that trajectory. Previous approaches typically either would predict likelihoods for a pre-determined set of goals (similar to the anchors used by [[YOLO]]) or predict the $xyz$ locations of the trajectory over time directly.

The highlights of the paper are:
- Use a transformer with motion query pairs (one for global intention localization and one for local movement refinement).
- Add a dense future prediction head to add information on the future interactions between the agent they are predicting for and other agents.
- They set a new SOTA result on [[WOMD|Waymo Open Motion Dataset]].

# Existing Approaches
Existing approaches fall into two main categories: goal-based methods and direct-regression methods.

### [[Goal Based Trajectory Prediction]]
These approaches predict the possible goals for an agent and then select a subset of the possible goals to predict a full trajectory for. [[Target-driven trajectory prediction|TNT]] selects anchors from the road maps and generates trajectories conditioned on the anchors. It then uses [[Non-maximum Suppression|NMS]] to select from these trajectories. [[End-to-end Trajectory Prediction from Dense Goal Sets|DenseTNT]] predicts a dense grid of possible goals and then uses a learned approach to select the final trajectories.

**Benefits**: trajectory uncertainty is reduced since you just need to complete possible trajectories from the start point to one of the selected goals.

**Downside:** they are very sensitive to the number of goal candidates. Fewer candidates will decrease performance and more candidates will increase computation and cost.

### Direct Regression
These approaches will directly predict the coordinates of parts of a trajectory over time. For instance, [[Scene Transformer]] predicts $T$ distinct time steps for each agent's trajectory and the agent's $xyz$ position at each step. You then link up the time steps to form the complete trajectory. The paper also mentions Multipath++ (Efficient information fusion and trajectory aggregation for behavior prediction) as another example.

**Benefits:** they can predict a broad range of agent behaviors since you aren't restricted to certain goal candidates.

**Downsides:** they tend to converge slowly since various motion modes are required to be regressed from the same agent feature without using any spatial priors (like hardcoding candidate goals to be near the center of a lane). They also tend to predict the most frequent modes of training data since these modes dominate the optimization of the agent feature.

# MTR Approach
MTR jointly optimizes two tasks:
??
- **Global intention localization:** Roughly identify the agent's intention.
- **Local movement refinement:** Perform local movement refinement to refine each intention's predicted.
<!--SR:!2024-01-14,104,270-->

This approach stabilizes the training process without depending on dense goal candidates. It also enables local refinement for each motion mode. Using mode-specific motion query pairs avoids training issues with predicting different motion modes.

### Motion Query Pairs
Instead of using goal candidates, MTR incorporates spatial intention priors by adopting a small set of learnable motion query pairs. Each motion query pair takes charge of trajectory prediction and refinement for a specific motion mode, which stabilizes the training process and facilitates better multimodal predictions.

Each motion query pair consists of two parts:
- A static intention query for global intention localization.
- A dynamic searching query for local movement refinement.

The motion query pair approach was motivated by [[DAB-DETR]] where the object queries of a transformer were considered as the positional embedding of a spatial anchor box. 

**Static Intention Queries:**
These queries are a learnable positional embedding of an intention point for generating a trajectory for a specific mode. This stabilizes the training process by using different queries for different modes and eliminates the dependency on dense goal candidates by requiring each query to take charge of a large region.

**Dynamic Searching Queries:**
They are initialized as the learnable embedding of the intention points and are responsible for retrieving fine-grained local features around each intention point.

They are dynamically updated according to the predicted trajectories and gather information from a deformable local region for iterative motion refinement.

### Dense Future Prediction Module
When predicting the motion for a particular agent (besides hero), it is important to consider the future interactions of all agents in the scene and not just the past.

To add future information on the other agents in the scene, they add an auxiliary regression head to densely predict future trajectory for each agent which are encoded as additional future context features for motion prediction of the agent they are predicting for.

# Architecture
![[mtr-architecture.png]]
> Figure 1: The architecture of MTR framework. (a) indicates the dense future prediction module, which predicts a single trajectory for each agent (e.g., drawn as yellow dashed curves in the above of (a)). (b) indicates the dynamic map collection module, which collects map elements along each predicted trajectory (e.g., drawn as the shadow region along each trajectory in the above part of (b)) to provide trajectory-specific feature for motion decoder network. (c) indicates the motion decoder network, where K is the number of motion query pairs, T is the number of future frames, D is hidden feature dimension and N is the number of transformer decoder layers. The predicted trajectories, motion query pairs, and query content features are the outputs from last decoder layer and will be taken as input to next decoder layer. For the first decoder layer, both two components of motion query pair are initialized as predefined intention points, the predicted trajectories are replaced with the intention points for initial map collection, and query content features are initialized as zeros.

### Input Representation
They represent both input trajectories and road maps as polylines.

For the motion prediction of a particular agent, they use an agent-centric approach that normalizes all inputs to the coordinate system centered at the agent.

A polyline encoder is then used to convert each polyline as an input token for the transformer encoder.

**Agent Representation**
The history state of $N_a$ agents is represented as $A_{\text {in }} \in \mathbb{R}^{N_a \times t \times C_a}$ where $t$ is the number of history frames, $C_a$ is the number of state information (ex. location, heading angle, and velocity), and zeros are padded at the positions of missing frames for trajectories.

**Road Representation:**
The road is represented as $M_{\text {in }} \in \mathbb{R}^{N_m \times n \times C_m}$ where $N_m$ is the number of map polylines, $n$ is the number of points in each polyline, and $C_m$ is the number of attributes of each point (ex. location and road type).

**Polyline Encoder**:
They use a [[PointNet]]-like polyline encoder to encode the agent features and map features to $A_{\mathrm{p}} \in \mathbb{R}^{N_a \times D}$ and $M_{\mathrm{p}} \in \mathbb{R}^{N_m \times D}$ respectively (with feature dimension $D$).

### Scene context encoder with local transformer encoder
The local structure of scene context is important for motion prediction. For example, the relation of two parallel lanes is important for modeling the motion of changing lanes, but adopting attention on global connected graph equally considers relation of all lanes.

They use [[Local Attention]] which better maintains the locality structure and is more memory efficient. Specifically, for a given query they will only use the keys and values from the $k$-closest polylines for each query polyline (found using $k$-nearest neighbor algorithm).

Specifically the $j$-th transformer encoder layer can be formulated as:
$$G^j=\text { MultiHeadAttn }\left(\text { query }=G^{j-1}+\mathrm{PE}_{G^{j-1}}, \text { key }=\kappa\left(G^{j-1}\right)+\mathrm{PE}_{\kappa\left(G^{j-1}\right)}, \text { value }=\kappa\left(G^{j-1}\right)\right)$$
where:
- $G^j$ is the j-th layers output ($G^0$ corresponds to the input).
- $G^0=\left[A_{\mathrm{p}}, M_{\mathrm{p}}\right] \in \mathbb{R}^{\left(N_a+N_m\right) \times D}$ (a concatenation of features for agents $N_a$ and map $N_m$ data with $D$ channels each).
- $\kappa(\cdot)$ denotes the $k$-nearest neighbor algorithm to find $k$ closest polylines for each query polyline.
- $\text{PE}$ are sinusoidal position encodings of the input tokens where they use the latest position for each agent and the polyline center for each map polyline.

The encoder produces agent features $A_{\text {past }} \in \mathbb{R}^{N_a \times D}$ and map features $M \in \mathbb{R}^{N_m \times D}$.

**Benefits of local attention:**
- You can handle larger scenes since you don't need to capture all pairwise polyline interactions.
- You have a better prior to pay attention to nearby lines.

### Dense future prediction for future interactions
The paper uses the encoded features $A_{\text {past }} \in \mathbb{R}^{N_a \times D}$ learned by the transformer encoder to predict the future trajectories and velocities of all agents by using it as the input to a regression head:
$$S_{1: T}=\operatorname{MLP}\left(A_{\text {past }}\right)$$

where $S_i \in \mathbb{R}^{N_a \times 4}$ includes the future position and velocity of each agent at time step $i$ and $T$ is the number of future frames to be predicted.

They predict each frame independently and then encode the trajectories using the same polyline encoder used to encode the original input representations.

They then concatenate the predicted future trajectories to the past trajectory information and use an MLP to produce a new summarized feature vector $A$ for all agent data ($A=\operatorname{MLP}\left(\left[A_{\text {past }}, A_{\text {future }}\right]\right)$).

> [!NOTE] Benefits of the dense scene prediction head
> Using this auxiliary task provides additional future context information to the decoder network which enables the model to make more scene-compliant future trajectories for the interested agent. Empirically, it improved the performance of multimodal motion prediction.


### Transformer Decoder with Motion Query Pair
They use a transformer-based motion decoder network for multimodal motion prediction. It contains stacked transformer decoder layers for iteratively refining the predicted trajectories with motion query pairs.

![[mtr-transformer-decoder-architecture.png]]

### Global Intention Localization
The paper uses a fixed set of intention points that each take charge of predicting trajectories for a certain motion mode. This helps stabilize training (you avoid learning queries that switch which mode of trajectory they represent) and helps narrow down the range of options the model needs to select from.

The $\mathcal{K}$ intention points $I \in \mathbb{R}^{\mathcal{K} \times 2}$ are found by using k-means clustering on the endpoints of ground-truth trajectories. This will tell you the $\mathcal{K}$ most common ways that a vehicle behaves. Note: the trajectories are in the agents frame of view so you don't need to worry about agents going opposite directions having different endpoints.

They only need a small number of queries instead of the densely goal candidates used in papers like [[End-to-end Trajectory Prediction from Dense Goal Sets|DenseTNT]].

### Local Movement Refinement