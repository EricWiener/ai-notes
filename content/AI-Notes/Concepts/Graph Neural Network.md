---
tags: [flashcards]
source: https://www.youtube.com/watch?v=zCEYiCxrL_0
summary:
aliases: [GNN]
---

[Medium Article](https://towardsdatascience.com/a-gentle-introduction-to-graph-neural-network-basics-deepwalk-and-graphsage-db5d540d50b3)
  
# Background Info
### Distributed vector representations
![[distributed-vector-representations.png]]
- With local representation we have a sparse vector of mostly zeroes with a single 1. However, we don't know how these items correlate or combine.
- With distributed representation we have denser vectors that show relationships between vectors (like in one-hot). Ex: both a banana and mango seem to share more features with each other than a dog.
- You can convert the 1-hot embedding to a distributed representation using an embedding layer (ex. word2vec). Ex: to convert a one-hot vector $\mathbb{I}_{w}$ to the distributed representation $\boldsymbol{r}$ , you would do $\boldsymbol{r}=E \mathbb{I}_{w}$ where $E$ is an embedding matrix with shape $D \times V$ (distributed representation size x vocabulary size). Typically $D$ would be much smaller than $V$.

# Graph Neural Networks
### Overview
![[screenshot-2022-03-04_07-33-36.png]]
- You start with an initial representation of the graph where each node contains information about it in a feature vector. How you create the graph is problem-dependent.
- You then pass the graph to a GNN.
- You end up with an output graph where now each node contains a feature vector (usually the same dimension) with information on how that node relates to the rest of the graph.
    - The resulting graph usually has the same structure (but doesn't have to).  
- You then pass the output of a GNN to be used for a task-specific goal.

### Neural Message Passing - Intuition
![[graph-neural-network-20220304074324309.png]]
- The example local-graph we are using is shown in the bottom-left. We want to pass messages from E and D to F. All nodes have their own feature vectors.

![[An Introduction to Graph Neural Networks_ Models and Applications 14-16 screenshot.png]]
- You are going to send a message from the neighbors (E, D) to F.
- Each message is a function of the ==features of the neighbor, the edge type, and the features of the current node==.
<!--SR:!2026-01-01,774,310-->


![[An Introduction to Graph Neural Networks_ Models and Applications 14-54 screenshot.png]]
- Once you get messages from your direct neighbors, you summarize the information.

![[An Introduction to Graph Neural Networks_ Models and Applications 15-3 screenshot.png]]
- You then combine the ==summarized information with the current state of the node==.
- You then update the node based on the combined information.
- $F$ had a state at time $t - 1$ and now it has a new state with information about itself and from its neighbors.
<!--SR:!2024-03-26,548,310-->

### Neural Message Passing - Math
![[An Introduction to Graph Neural Networks_ Models and Applications 16-23 screenshot.png]]
- Prepare "Message" (shown in black) is a function $f_t$ that takes the representation of the current node $n$ at timestamp $t-1$, the edge information $k$, and the representation of the neighboring node $n_j$ at timestamp $t-1$.
- Summarize received information (blue) computes the previous function for all neighbors of $n$. You can use many different ways to combine the information (ex. summation, max, etc.).
- $\boldsymbol{h}_{t}^{n}=q\left(\boldsymbol{h}_{t-1}^{n}, \boldsymbol{x}\right)$ (orange) takes the features $h_{t-1}^n$ of the current node $n$ at time $t-1$ and takes the summarized features and then produces the next representation of the current node via the function $q$.

### Updating all nodes
![[An Introduction to Graph Neural Networks_ Models and Applications 19-26 screenshot.png]]
- You iteratively will update all the nodes based on their immediate neighbors.
- Each time step you will pass messages again and the longer you run the algorithm, the more information from distant nodes can propogate.
- All nodes update their state in parallel.
- You usually don't "unfold" for too many steps - around 10 (since you will later have to backpropagate).

![[An Introduction to Graph Neural Networks_ Models and Applications 19-56 screenshot.png]]
- You will always use just the immediate neighbors regardless of what time iteration you are on.

### Loss + Backprop
![[An Introduction to Graph Neural Networks_ Models and Applications 24-47 screenshot.png]]
- At each time step you will compute a loss (ex. shown here is you make a binary node classification and then compute binary cross entropy loss).
- You then backpropgate and update all your parameters.

# Gated GNNs
![[An Introduction to Graph Neural Networks_ Models and Applications 28-6 screenshot.png]]
- Prepare message: $E_{k} \boldsymbol{h}_{t-1}^{n_{j}}$. Where $k$ is the type of the edge. You have a weight matrix for each type of edge.
- Summarize received information: $\boldsymbol{m}=\sum_{\forall n_{j}: n \rightarrow n_{j}} E_{k} \boldsymbol{h}_{t-1}^{n_{j}}$
    - This just sums up all the computed messages
- Update state: $\boldsymbol{h}_{t}^{n}=\operatorname{GRU}\left(\boldsymbol{h}_{t-1}^{n}, \boldsymbol{m}\right)$ here the ==current state and summed up messages== are passed into a GRU. This will then give you the node's features at the next time step.
<!--SR:!2027-03-10,1399,330-->

# Graph Convolutional Networks
![[An Introduction to Graph Neural Networks_ Models and Applications 29-34 screenshot.png]]
- These resemble convolutions (hence the name)
- $\sum_{\forall n_{j}: n \rightarrow n_{j}} \boldsymbol{h}_{t-1}^{n^{j}}$ sums up the states of the neighbors
- $\boldsymbol{h}_{t-1}^{n}+\sum_{\forall n_{j}: n \rightarrow n_{j}} \boldsymbol{h}_{t-1}^{n^{j}}$ sums up the state of the current node + the states of the neighbors
- $W_t$ is the weight matrix
- $\frac{1}{\text { numneighbors }+1}$ normalizes
- $\sigma$ applies sigmoid.
- This implementation doesn't use a different weight matrix per edge type. Also is permutation equivariant.
- No longer uses GRU.

# Backwards Edges
![[An Introduction to Graph Neural Networks_ Models and Applications 30-42 screenshot.png]]
- To propogate information from nodes that don't have any outgoing edges (ex. node D), you can add backward edges.
- For this, you add an outgoing edge for every incoming edge to make sure information can flow in all directions.

# Expressing operations mathematically
### Background: adjacency matrix
![[An Introduction to Graph Neural Networks_ Models and Applications 33-54 screenshot.png]]
- Here you have a 1 if node $i$ has an outgoing connection to node $j$ (where $i$ is the row and $j$ is the column).

![[An Introduction to Graph Neural Networks_ Models and Applications 34-40 screenshot.png]]
- You can then multiply the adjacency matrix by a vector with the corresponding nodes ($N$).
- Then, when you do $A \cdot N$, you will end up with:
$$\left[\begin{array}{c}
0 \\
a \\
a+b
\end{array}\right]$$
- This tells us that:
    - Node $a$ didn't receive any information (denoted by 0).
    - Node $b$ received information from node $a$
    - Node $c$ received information from node $a$ and $b$ (denoted by $a + b$).
- You will have a different adjacency matrix for each edge type.

### Gated GNN as Matrix Operation
![[An Introduction to Graph Neural Networks_ Models and Applications 35-58 screenshot.png]]
- Node states: $H_t$ is a matrix that contains the hidden states for all the nodes. Each row corresponds to a node and has length of $D$ (the dimension of the features). $H_t$ therefore has dimensions (num_nodes x $D$).
- Messages to-be sent: you can compute all the messages to send for a particular edge type ($k$) by multiplying your weight matrix $E_k$ by the node state matrix $H_t$. $M^k_t$ will have dimensions (num_nodes x $M$) where $M$ is the number of messages.
- Received messages: you can compute the messages that are actually received via $\sum_{k} A_k M_{t}^{k}$. This sums over all the edge types ($k$). For each edge type, it will multiply the adjacency matrix for that edge type ($A_k$) by the messages of that edge type. After summing over all edge types, you will end up with $R_t$ which is dimension (num nodes x $M$).
    - Note $A$ in the above picture should be $A_k$.
- Update: you then pass the current node states ($H_t$) and all the messages that are passed ($R_t$) into your GRU to compute the next state.

Q: doesn't $M$ and num_nodes change depending on the graph?

# Sample Applications
- You can use GNNs for analyzing molecular structures
- You can use GNNs for analyzing code (syntax trees)

![[An Introduction to Graph Neural Networks_ Models and Applications 47-3 screenshot.png]]
- Ex: trying to auto-complete using graph of the code.

# Special Cases
![[An Introduction to Graph Neural Networks_ Models and Applications 48-36 screenshot.png]]
- You can re-express a convolutional layer as a graph neural network by connecting each pixel to the neighboring pixels (and to itself).

