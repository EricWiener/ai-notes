---
tags: [flashcards, eecs492, eecs498-dl4cv]
source:
summary:
---

> [!note]
> Markovian assumption: Where you go next depends only on where you go right now. Where are are at time t+1 only depends on time t.
> 

## MDP: Overview

- Environment: fully observable, stochastic
- Markovian transition model: The probability of the next state is based on the current state and the action: $P(s'|s, a)$
- The rewards add up over time

**Description:**
- **State** (initial state: $s_0$)
- Set of **actions**: ACTIONS(S) can execute in certain state
- **Transition** model: $P(s'|s,a)$
- **Reward** function: $R(s)$
- **Discount** factor: $\gamma$ tells us how much to value current vs. future rewards

**Policy**: specifies a probability distribution of what action an agent should take in any state. $\pi$ and $\pi(s)$

**Trajectory:** following a specific policy ($\pi$) will produce a trajectory: $s_0, a_0, r_0, s_1, a_1, r_1, ...$.

**Goal:** find a really good policy $\pi *$ that maximizes the total discounted reward $\sum_t \gamma^t r_t$.

**Terms:**
- Complete policy: the agent knows what to do in any possible state
- Optimal policy:
    - policy that yields the highest expected utility. Denoted $\pi*$
    - This is a simple reflex agent → just does something based on what state it is in.
- Environments:
    - Finite horizon: has a limited amount of time
        - Behavior is non-stationary
    - Infinite horizon: unlimited time to do whatever it wants
        - Behavior is stationary

## MDP: Algorithm
When you first start, the environment will choose an initial state $s_0$. The probability of this state being chosen is $p(s_0)$.

Then, you will repeat the following from $t=0$ until you are done:

- The agent will select an action $a_t$. This action is selected from the policy $\pi$ of $a$ conditioned on the current state $s_t$: $\pi(a | s_t)$.
- The environment will sample the reward $r_t$ to provide from the distributions of all rewards $r$ based on the current state $s_t$ and the action the agent chose $a_t$: $R(r|s_t, a_t)$.
- The environment will sample the next state $s_{t+1}$ using the transition function $P$ from the distribution of states $s$ conditioned on the previous state and the action: $P(s|s_t, a_t)$.
- The agent receives reward $r_t$ and the next state $s_{t+1}$

## MDP: Grid World
A grid world is the canonical example used to explain MDP problems.

![[AI-Notes/Reinforcement Learning/markov-decision-process-(mdp)-srcs/Screen_Shot.png]]

![[AI-Notes/Reinforcement Learning/markov-decision-process-(mdp)-srcs/Screen_Shot 1.png]]

## MDP: Optimal Policy
You want to find the optimal policy $\pi *$ that maximizes the discounted sum of rewards. However, there is randomness (initial state, transition probabilities, rewards are all random). Therefore, we need to **maximize the expected sum of rewards**, since we don't know what the actual sum of rewards will be.
![[AI-Notes/Reinforcement Learning/markov-decision-process-(mdp)-srcs/Screen_Shot 2.png]]

We now need to maximize the expected reward. a ~ b means that a comes from b.

## MDP: Value Function and Q Function
The **value function** tells us the expected cumulative reward we can expect to get if we follow a certain policy starting from state $s$.

$$
V^\pi (s) = \mathbb{E}[\sum_{t \geq 0} \gamma^t r_t | s_0 = s, \pi ]
$$

The **Q function** is slightly modified version of the value function. It is the expected cumulative reward from first taking action $a$ in state $s$, and then following the policy:

$$
Q^\pi (s, a) = \mathbb{E}[\sum_{t \geq 0} \gamma^t r_t | s_0 = s, a_0 = a, \pi ]
$$

Algorithms will either use the value function or Q function, but it is more common to use the Q function.

> [!note]
> For a given policy $\pi$, the value function tells us how good the state is. The Q function tells us how good a state-action pair is.
> 

The **optimal Q function** $Q^*(s, a)$ is Q-function for the optimal policy $\pi^*$. It gives the max possible future reward when taking action $a$ in state $s$. 

$$
Q^* (s, a) = \max_\pi \mathbb{E}[\sum_{t \geq 0} \gamma^t r_t | s_0 = s, a_0 = a, \pi ]
$$

You take the maximum of all the possible Q functions. Q* encodes the optimal policy: $\pi^*(s) = \argmax_{a'} Q(s, a')$. The policy tells us the best action to take for every possible state. The Q policy tells us for every state and every action, which is the best possible action to take. For every state, you can just check over all the actions to find which is best. **Because the optimal policy can be derived from the Q function, we no longer need the optimal policy - we can just use the Q function.**

## MDP: Bellman Equation

The **Bellman Equation** provides a recurrence relationship for the Q function. The Q function tells us the best reward we can get if we start in state $s$ and take a certain action $a$. After taking that action $a$, we then need to act optimally. To act optimally, you need to take the next action $a'$ to maximize $\max_{a'} Q^*(s', a')$.

Therefore, the Bellman Equation gives us a recurrence relationship that Q* must satisfy:

$$
Q^* (s, a) = \mathbb{E}_{r, s'}[r + \gamma \max_{a'}Q^*(s', a')]
$$

- We first take an initial action $a$ and end up in state $s'$.
- This initial action gives us a reward $r$
- After this, we need to take the action that maximizes the Q function. We add the $\gamma$ term because we discount the future actions.

> [!note]
> Therefore, if we find a function $Q(s, a)$ that satisfies the Bellman Equation, then it must be $Q^*$.
> 

## MDP: Value Iteration

We can solve for the optimal policy by repeatedly making improvements to the $Q$ function. We start with a random $Q$ and use the Bellman Equation as an update rule:

$$
Q_{i+1} (s, a) = \mathbb{E}_{r, s'}[r + \gamma \max_{a'}Q_i(s', a')]
$$

You keep iterating over optimal policies. Each time you iterate, the policy will become the max of the previous policies, so it will keep improving. 

**Fact:** $Q_i$ ****converges to $Q^*$ as $i \rightarrow \infty$. 

However, we have a problem. In order to evaluate the best Q function at each time step, we need to calculate the Q value for every possible (state, action) pair $Q(s, a)$. **This is impossible if the exploration space is infinite.**

## MDP: Deep Q-Learning

Because it is impossible to calculate $Q(s, a)$  for an infinite space. We can approximate $Q(s, a)$ with a neural network and use the Bellman Equation as the loss.

**Inputs:** The network will receive $Q(s, a; \theta)$

- A state $s$
- An action $a$
- Weights of the network $\theta$

**Outputs:** The network will output an approximation for $Q^*(s, a)$

**Training:**

If the neural network was working correctly, the optimal policy approximation should satisfy the Bellman Equation. In order to create a loss function, we need to find an approximation for what the network should be prediction

For a particular state and a particular action $(s, a)$, you can sample a bunch of potential next states $s'$ and actions $a'$ using the network's predictions to give us a target $y$ for what the network should predict.

$$
y_{s, a, \theta} = \mathbb{E}_{r, s'}[r + \gamma \max_{a'}Q(s', a'; \theta)]
$$

We can then formulate a loss for training $Q$ to try to get it to be as close as possible to $y_{s, a, \theta}$. This is looking ahead at what a good target prediction is and then trying to get the network to predict this.

$$
L(s, a) = (Q(s, a; \theta) - y_{s, a, \theta})^2
$$

Hopefully the network will eventually learn an optimal $Q^*$ policy. However, we have a **nonstationary problem.** As the network learns, it's predictions for the policies will change. Therefore, the target $y_{s, a, \theta}$ will also change over time and depends on the current weights $\theta$.

Additionally, we need a way to figure out which data points we should sample, how we form minibatches, etc.

## Policy Gradients

For Q Learning, we train a network $Q_\theta(s, a)$ top estimate the future rewards for every (state, action) pair. However, sometimes it is easier just to learn a direct mapping from states to actions rather than indirectly learning the mapping through the Q function.

Ex. if we wanted to pick up a bottle, you would just want to move your hand till it touches the bottle, close your fingers, and then lift your hand.

**Policy gradients:** train a network $\pi_\theta(a |s)$ that takes state as an input and gives a distribution over which action to take in that state. 

**Objective function:** we want to maximize the expected sum of future rewards. We can calculate the expected future rewards when following policy $\pi_\theta$ with:

$$
J(\theta) = \mathbb{E}_{r \sim p_\theta} [\sum_{t \geq 0}\gamma^tr_t]
$$

You can find the optimal policy by maximizing $J(\theta)$: $\theta^* = \argmax_\theta J(\theta)$. We can do this using gradient ascent. However, we don't know how to compute $\frac{dJ}{d\theta}$ since we can't differentiate through the world.

### Policy Gradients: REINFORCE Algorithm

We can derive a gradient we are able to optimize. 

$$
J(\theta) = \mathbb{E}_{x \sim p_\theta} [f(x)] \\ \frac{dJ}{d\theta} = \mathbb{E}_{x \sim p_\theta} [f(x)\sum_{t\geq0} \frac{\partial}{\partial \theta} \log \pi_\theta(a_t|s_t)]
$$

- $\mathbb{E}_{x \sim p\theta}$ is the expected value evaluated over the trajectories $x$ that are sampled by following the policy $\pi_\theta$ in the environment.
- $f(x)$ is the reward we get from the trajectory $x$
- $\frac{\partial}{\partial \theta} \log \pi_\theta(a_t|s_t)$ is the gradient of the predicted action scores with respect to the model weights. You backprop through the model $\pi_\theta$ that the network predicts.

**Algorithm:**

Initialize random weights $\theta$

1. Run the policy $\pi_\theta$ multiple times in the environment to collect trajectories $x$ and rewards $f(x)$
2. Compute $\frac{\partial J}{\partial \theta}$ (the derivative of the reward with respect to the policy)
3. Perform a gradient ascent step on $\theta$
4. Repeat

**Intuition:**

- When $f(x)$ is high, you get a large reward. You want to increase the probability of the actions we took.
- When $f(x)$ is low, you get a small reward. You want to decrease the probability of the actions we took.

This is much harder to train than with regular gradient descent with supervised learning. Also, all actions become more likely if the reward is high (this is the **credit assignment problem**). You can use a **value function** to give you the expected future reward starting from a certain state. This can help figure out what states and actions are good.

## Other Models

**Actor-Critic:** train an **actor** that predicts actions (like policy gradient) and a **critic** that predicts the future rewards we get from taking those actions (like Q-Learning).

**Model-Based:**  So far, we haven't tried to explicitly model the transition function. A model-based approach tries to learn a model of the world's state transition function $P(s_{t+1}|s_t, a_t)$ and then use planning through the model to make decisions.

For instance, if the states of the environment are images, you want to predict the future (future images) after you perform an action.

**Imitation Learning:** gather data about how experts perform in the environment. Then try to learn a function to imitate what they do. This is a form of supervised learning.

**Inverse Reinforcement Learning:** you collect data about what experts do. Then, you try to learn a reward function that they seem to be optimizing. You then use reinforcement learning to to try to optimize using this reward function.

**Adversarial Learning:** learn to fool a discriminator that predicts actions as real/fake.

## Case Study: AlphaGo

Google built a system using reinforcement learning that beat the reigning AlphaGo champion.

**AlphaGo:**

- Used imitation learning + tree search + reinforcement learning
- Beat 18 time world champion Lee Sedol (retired because AI got too good).

**AlphaGo Zero:**

- Simplified version of AlphaGo
- No longer used imitation learning
- Beat the #1 player

**Alpha Zero:**

- Generalized to other games: Chess and Shogi

**MuZero:**

- Used model-based reinforcement learning

## Stochastic Computation Graphs

We can use reinforcement learning to train neural networks with non-differentiable components. For example, you might have a multi-part network, where the first part of the network predicts a distribution over what part of the network to use next. You can then sample from this distribution to decide the next part of the network to use. (Note: you shouldn't actually do this).

The random sampling isn't differentiable, but you can train this with reinforcement learning.
![[AI-Notes/Reinforcement Learning/markov-decision-process-(mdp)-srcs/Screen_Shot 3.png]]
Dummy example showing non-differentiable network using RL.

Another example of this being done is in the paper we looked at previously for image captioning using attention. The top-row shows **soft-attention** being used, where a weighted average of image features from different spatial positions was used. 

![[AI-Notes/Reinforcement Learning/markov-decision-process-(mdp)-srcs/Screen_Shot 4.png]]

In the bottom-row, **hard attention** is used. Here, you select features from only one spatial location, instead of doing a weighted average. You can train this with **policy gradient**.