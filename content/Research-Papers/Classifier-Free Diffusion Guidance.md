---
tags: [flashcards]
source: https://arxiv.org/abs/2207.12598
summary:
---

[YouTube Overview](https://youtu.be/c1GwVg3lt1c?t=225)

Classifier guidance combines the score estimate of a diffusion model with the gradient of an image classifier and thereby requires training an image classifier (on noisy images) separate from the diffusion model. It also raises the question of whether guidance can be performed without a classifier. We show that guidance can be indeed performed by a pure generative model without such a classifier: in what we call classifier-free guidance, we jointly train a conditional and an unconditional diffusion model, and we combine the resulting conditional and unconditional score estimates to attain a trade-off between sample quality and diversity similar to that obtained using classifier guidance.

# Classifier-free guidance
Classifier-free guidance doesn't require pre-training a classifier on noisy images. Instead, you replace the label $y$ in a class-conditional diffusion model $\epsilon_\theta\left(x_t \mid y\right)$ with the ==null label $\emptyset$== with a fixed probability during training (typically 10-20%).
<!--SR:!2024-06-22,264,308-->

During sampling, the output of the model is extrapolated further in the direction of the class conditioned model $\epsilon_\theta\left(x_t \mid y\right)$ and away from $\epsilon_\theta\left(x_t \mid \emptyset\right)$ with the following:
$$\hat{\epsilon}_\theta\left(x_t \mid y\right)=\epsilon_\theta\left(x_t \mid \emptyset\right)+s \cdot\left(\epsilon_\theta\left(x_t \mid y\right)-\epsilon_\theta\left(x_t \mid \emptyset\right)\right)$$
where $s \geq 1$ is the guidance scale.

### Geometric Interpretation
A geometric interpretation for a single point of what $\hat{\epsilon}_\theta\left(x_t \mid y\right)=\epsilon_\theta\left(x_t \mid \emptyset\right)+s \cdot\left(\epsilon_\theta\left(x_t \mid y\right)-\epsilon_\theta\left(x_t \mid \emptyset\right)\right)$ is accomplishing is below:
![[classifier-free-guidance-diagram.excalidraw|700]]
- $\epsilon_\theta\left(x_t \mid \emptyset\right)$ is the output of the model conditioned on the null label
- $\epsilon_\theta\left(x_t \mid y\right)$ is the output of the model conditioned on the class label
- $\epsilon_\theta\left(x_t \mid y\right)-\epsilon_\theta\left(x_t \mid \emptyset\right)$ is the difference between the conditioned and unconditioned points.
- $s \cdot (\epsilon_\theta\left(x_t \mid y\right)-\epsilon_\theta\left(x_t \mid \emptyset\right))$ applies a scaling term $\geq 1$ to the difference vector.
- You then add this scaled difference term to the unconditioned point to get your final point $\hat{\epsilon}_\theta\left(x_t \mid y\right)$.

### Benefits
Classifier-free guidance has two appealing properties:
??
- It allows a single model to leverage its own knowledge during guidance, rather than relying on the knowledge of a separate (and sometimes smaller) classification model.
- It simplifies guidance when conditioning on information that is difficult to predict with a classifier (such as text).
<!--SR:!2024-05-21,232,270-->