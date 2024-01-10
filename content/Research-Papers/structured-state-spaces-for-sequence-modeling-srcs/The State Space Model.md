---
tags:
  - flashcards
source: https://en.wikipedia.org/wiki/State-space_representation
summary: The state space model tells you how the state of a system changes over time and allows you to calculate the output of a system at a given time given the input to the system.
---
The State Space Model is defined by:
$$\begin{array}{c}{{\dot{\mathbf{x}}(t)=\mathbf{Ax}(t)+\mathbf{B}\mathbf{u}(t)}}\\ {{\mathbf{y}(t)=\mathbf{Cx}(t)+\mathbf{D}\mathbf{u}(t)}}\end{array}
$$
$\dot{\mathbf{x}}(t)$ is the derivative of the state space $\mathbf{x}(t)$ with respect to time and is defined by ${\dot{\mathbf{x}}}(t):={\frac{d}{d t}}\mathbf{x}(t)$. It tells you how the state vector changes with time.

The first equation is the state equation and it tells you how the state $\mathbf{x}$ changes over time. The derivative $\dot{\mathbf{x}}(t)$ tells you the rate of change of the state variables and it is influenced by the current state $\mathbf{x}$ and the input to the system at time $t$, $\mathbf{u}(t)$.

The second equation is the output equation. It describes how the state $\mathbf{x}$ and the input $\mathbf{u}(t)$ are used to generate the output $\mathbf{y}(t)$.

> [!NOTE] Other forms of the state space model exist
> The form shown above is specifically the continuous time invariant form where $A, B, C, D$ don't change over time.
> 
> For reference, another form, the continuous time-variant form looks like:
> $$\begin{aligned}& \dot{\mathbf{x}}(t)=\mathbf{A}(t) \mathbf{x}(t)+\mathbf{B}(t) \mathbf{u}(t) \\& \mathbf{y}(t)=\mathbf{C}(t) \mathbf{x}(t)+\mathbf{D}(t) \mathbf{u}(t)\end{aligned}$$

[This](https://youtu.be/hpeKrMG-WP0) is a very good video on the State-Space Equation.
![[Research-Papers/structured-state-spaces-for-sequence-modeling-srcs/assets/The State Space Model/image-20240103135955390.png]]

[Source](https://youtu.be/hpeKrMG-WP0?t=351)

- $\mathbf{u}(t)$ is the input signal.
- $\mathbf{x}(t)$ is the state vector. This is usually not directly measurable for most times (you may know it at a specific time but not at all times). It is a vector that contains all the state variables.
- $\mathbf{y}(t)$ is the output signal. It is a 
- $\dot{\mathbf{x}}(t)$ is the derivative of the state space $\mathbf{x}(t)$ with respect to time. It is a linear combination of the current state and the inputs to the system.
- $A$ tells us how the internal states are all connected to each other.
- $B$ tells us which states are affected when inputs enter into the system.
- $C$ tells us how the states are combined to get the outputs.
- $D$ allows us to bypass the system entirely and have the inputs directly affect the outputs (like a residual connection).

> [!QUESTION] Why do we care about $\dot{\mathrm{x}}(t)$ if it's not used when calculating $y(t)$?
> When calculating $y(t)$ you need to know what $x(t)$ is. However, $x(t)$ is typically not directly observable so you need to integrate using $\dot{\mathrm{x}}(t)$ in order to find $x(t)$ at a given time.
### Example of using a state space equation
Let's consider a simple example with a single input, single output (SISO) system. Let's say we have a system represented by the following state space equations:

1. State equation: `x'(t) = Ax(t) + Bu(t)`
2. Output equation: `y(t) = Cx(t) + Du(t)`

Let's assume `A = -1`, `B = 2`, `C = 3`, and `D = 0`. 

So, our system becomes:
1. `x'(t) = -x(t) + 2u(t)`
2. `y(t) = 3x(t)`

Let's also assume that the input to the system `u(t) = 1` for all `t` (a constant input), and the initial state `x(0) = 0` (the system starts at rest).

The solution to the state equation `x'(t) = -x(t) + 2` is `x(t) = 2 - 2e^(-t)`. You can verify this by taking the derivative of `x(t)` and substituting back into the state equation.

Then, substituting `x(t)` into the output equation gives `y(t) = 3x(t) = 6 - 6e^(-t)`.

So, given our initial conditions and input, the system state `x(t)` and output `y(t)` evolve over time according to these equations.

> [!question] Why did you need to integrate x'(t) if you could have just done `y(t) = Cx(t) + D u(t)` which becomes `y(t) = 3x(t) + 0 = 3x(t)`. You could have just then done `3x(t)`
> Your observation is correct. However, the state `x(t)` is not known directly from the input `u(t)`. The state equation `x'(t) = -x(t) + 2u(t)` describes how the state `x(t)` evolves over time based on the input `u(t)`. To find the value of `x(t)` at any given time, we need to solve this differential equation, which typically involves integration.
> 
> Once we know `x(t)`, we can substitute it into the output equation `y(t) = Cx(t) + Du(t) = 3x(t)`. Without knowing `x(t)`, we can't calculate `y(t)`.
> 
> In control systems, the state `x(t)` often represents internal variables of the system that we don't directly control or measure. However, these states influence the output `y(t)`, which is why we need to calculate it. 
> 
> So, the process is: use the state equation to find `x(t)` from `u(t)`, and then use the output equation to find `y(t)` from `x(t)` and `u(t)`.

# Different Interpretations
### The Continuous-time Representation
$$\begin{array}{c}{{\dot{\mathbf{x}}(t)=\mathbf{Ax}(t)+\mathbf{B}\mathbf{u}(t)}}\\ {{\mathbf{y}(t)=\mathbf{Cx}(t)+\mathbf{D}\mathbf{u}(t)}}\end{array}$$
Below is a block diagram of the continuous time representation of the state space model ([source](https://en.wikipedia.org/wiki/State-space_representation)):
![[Research-Papers/structured-state-spaces-for-sequence-modeling-srcs/the-state-space-model-srcs/the-state-space-model-20231228185115359.png]]

The diagram explained from left to right is:
- $u$ is the input to the system
- You calculate $\mathbf{Ax}(t)+\mathbf{B}\mathbf{u}(t)$ to get $\dot{\mathbf{x}}(t)$.
- You integrate $\dot{\mathbf{x}}(t)$ to get $\mathbf{x}$.
- You calculate ${\mathbf{y}(t)=\mathbf{Cx}(t)+\mathbf{D}\mathbf{u}(t)}$ to get the output

You can think of this as a function mapping $u(t)\mapsto y(t)$ parameterized by $\mathbf{A, B, C, D}$. However, for applications of machine learning, you won't actually get the input data in a continuous form since most signals (ex. audio) are discretized via sampling (ex. sampling sound at a specific frequency).

### Computing the SSM with Recurrence
In real world applications, data is discrete. Therefore, instead of representing the input and outputs as continuous values via $u(t)$ and $x(t)$, you can represent them as discrete $u_k$ and $y_k$ and define a linear recurrence that operates one step at a time (vs. integrating with respect to time).
$$\begin{array}{c}{{x_{k}=\bar{A} x_{k-1}+\bar{B} u_{k}}}\\ {{y_{k}=\bar{C}x_{k}+\bar{D}u_{k}}}\end{array}$$
The parameters of the model are represented by $\bar{A}, \bar{B}, \bar{C}, \bar{D}$ to signify these are different from the parameters used in the continuous version of the equation. This ends up looking like a simple [[AI-Notes/Concepts/RNN, LSTM, Captioning|RNN]] where the inputs $u_k$ are processed one at a time, are used to update the hidden state $x_k$, and then you calculate the output $y_k$ at each state.
![[Research-Papers/structured-state-spaces-for-sequence-modeling-srcs/the-state-space-model-srcs/the-state-space-model-20231228190310071.png|500]]
The diagram explained with reference to $x_k$:
- You take your previous hidden state $x_{k-1}$ and compute $\bar{A}x_{k-1}$.
- You then calculate your new hidden state with $x_k$ with $\bar{A}x_{k-1}$ and $\bar{B}u_k$ where $u_k$ is the input at step $k$
- You output $y_k$ for step $k$ based on $\bar{C}x_k$ and $\bar{D}u_k$ (note that the $\bar{D}u_k$ operation is shown on the left side only).

### Computing the state space with convolutions
It's possible to compute linear recurrences in parallel using convolutions. To see this, we first expand the first couple states of the recurrence above:
$$x_{0}=\overline{{{B}}}u_{0}\quad x_{1}=\overline{{{A}}}\overline{{{B}}}u_{0}+\overline{{{B}}}u_{1}\quad x_{2}=\overline{{{A}}}^{2}\overline{{{B}}}u_{0}+\overline{{{A}}}\overline{{{B}}}u_{1}+\overline{{{B}}}u_{2}\quad...$$



