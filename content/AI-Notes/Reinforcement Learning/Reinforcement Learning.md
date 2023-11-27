Files: 498_FA2019_lecture21_(2).pdf
Tags: EECS 498

Reinforcement learning is separate from both unsupervised and supervised learning. You build **agents** that can interact with the world. It performs **actions** in an **environment** and receives rewards for those actions. Your goal is to take actions that maximize reward.

We control the agent, but we don't have control over what happens in the environment. The environment provides the agent with a **state**. The state provides information about what is going on in the world (can be noisy and incomplete). The agent communicates back to the environment by performing an **action** based on the state.  The environment will send back a **reward** that tells the agent how well it is doing.

The environment will change over time as a result of the actions and the agent will change over time as it learns based on the rewards.

![[The environment gives an agent a state, the agent gives an action, and the environment gives a reward.]]

The environment gives an agent a state, the agent gives an action, and the environment gives a reward.

**Examples:**

- Cart-Pole Problem: balancing a pole on top of a movable cart
- Robot Locomotion: make a robot learn to move forward
- Atari Games: play a game with the highest score
- Go: play against an opponent in Go

## Supervised Learning vs. Reinforcement Learning

![[Showing supervised learning in a similar diagram]]

Showing supervised learning in a similar diagram

We can think of supervised learning in a similar lens as reinforcement learning:

- The training batch $x_t$ is the state
- The action $y_t$ is the prediction by the model
- The reward is the loss $L_t$

However, supervised learning and reinforcement learning are very different.

**Difference #1: Rewards and state transitions may be random**

There can be a lot of noise. The states can be incomplete. The rewards can be random (you might get different rewards in different time steps). The transitions can be noisy.

**Difference #2: Reward** $r_t$ **may not depend directly on action** $a_t$

Rewards at one time-step might be a result of previous actions (reaching the end of the maze is dependent on all previous moves, not just the final move to the end). This issue is known as **credit assignment** (you don't know what action deserves credit)

**Difference #3: You can't backpropagate through the world**

This problem is non-differentiable. You can't compute $\frac{dr_t}{da_t}$.

**Difference #4: Nonstationary**

The agent is moving through the world. What the agent experiences depends on how it acts. If it learns to move around, it will explore the world and see new data. The data the agent is exposed to is a function of how well it has learned to interact with the environment.

GANs also suffer from this issue. The discriminator starts seeing different data as the generator gets better.

> [!note]
> Reinforcement is a very hard problem. It is much harder than supervised learning.
> 

## Imitation Learning

Imitation learning is supervised learning applied to learn **policies**. For each state $s_i$, you watch what an expert does and get an action $a_i$.

This is what early game playing agents did to learn. This is also what Donkeycars do.

![[AI Notes/Reinforcement Learning/Reinforcem/Screen_Shot 2.png]]