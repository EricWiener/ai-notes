---
tags: [flashcards]
source: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
summary: the Diffusion Kernel is used by diffusion models to sample a noised image at a particular timestep.
---

$q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right)$ is the Diffusion Kernel and is used in the forward diffusion process to go from a less noisy image $x_{t-1}$ to a noisier image $x_t$. It is defined as $q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right)=\mathcal{N}\left(\mathbf{x}_{\mathbf{t}} ; \sqrt{1-\beta_t} \mathbf{x}_{\mathbf{t}-\mathbf{1}}, \beta_t \mathbf{I}\right)$. We can sample from it using the [[The Reparameterization Trick]].

### Applying the reparameterization trick to sample the next time step
If we have the image at time step $t - 1$, we can get the slightly noisier image at step $t$ using the reparamterization trick:
$$\begin{aligned}
& \mathrm{q}\left(\mathrm{x}_t \mid x_{t-1}\right)=\mathcal{N}\left(x_t, \sqrt{1-\beta_t} x_{t-1}, \beta_t I\right) \\
&=\sqrt{1-\beta_t} x_{t-1}+\sqrt{\beta_t} \epsilon
\end{aligned}$$
We can then re-write this using a new scalar $\alpha_t = 1 - \beta_t$:
$$\begin{aligned}
&=\sqrt{\alpha_t} x_{t-1}+\sqrt{1-\alpha_t}\epsilon 
\end{aligned}$$
### Modifying the diffusion kernel to go from time step 0 directly to time step $t$
We went from time step $t - 1$ to time step $t$ using $q(x_t | x_{t-1}) = \sqrt{\alpha_t} x_{t-1}+\sqrt{1-\alpha_t}\epsilon$. 

You can then extend this to earlier time steps and go from $t-2$ directly to $t$ by just chaining the $\alpha$'s:
$$\begin{aligned}
&=\sqrt{\alpha_t \alpha_{t-1}} x_{t-2}+\sqrt{1-\alpha_t \alpha_{t-1}} \epsilon
\end{aligned}$$
this comes from combining the two steps:
- $x_{t-1}=\sqrt{\alpha_{t-1}} x_{t-2}+\sqrt{1-\alpha_{t-1}}\epsilon_{t-2}$
- $x_t =\sqrt{\alpha_t} x_{t-1}+\sqrt{1-\alpha_t}\epsilon_{t-1}$

We can replace $x_{t-1}$ in the second equation with the first equation:
$$x_t =\sqrt{\alpha_t}(\sqrt{\alpha_{t-1}} x_{t-2}+\sqrt{1-\alpha_{t-1}}\epsilon_{t-2})+\sqrt{1-\alpha_t}\epsilon_{t-1}$$
We can then simplify:
$$\begin{aligned} x_t = \sqrt{\alpha_t\alpha_{t-1}} x_{t-2}+\sqrt{\alpha_t}\sqrt{1-\alpha_{t-1}}\epsilon_{t-2} +\sqrt{1-\alpha_t}\epsilon_{t-1}
& \\ x_t = \sqrt{\alpha_t\alpha_{t-1}} x_{t-2}+\sqrt{\alpha_t-\alpha_t\alpha_{t-1}}\epsilon_{t-2} +\sqrt{1-\alpha_t}\epsilon_{t-1}
\end{aligned}$$
We can then merge the two Gaussian distributions $\epsilon_{t-2}$ and $\epsilon_{t-2}$. These both have zero mean and unit variance, so their updated variance is just given by the scalar applied to them. We therefore need to merge the two distributions $\epsilon_{t-2} = \mathcal{N}\left(0, (\sqrt{1 - \alpha_t})^2 I\right)$ and $\epsilon_{t-1} = \mathcal{N}\left(0, (\sqrt{\alpha_t-\alpha_t\alpha_{t-1}})^2 I\right)$. To merge two Gaussians with different variances, $\mathcal{N}\left(\mathbf{0}, \sigma_1^2 \mathbf{I}\right)$ and $\mathcal{N}\left(\mathbf{0}, \sigma_2^2 \mathbf{I}\right)$,  you add their means and the new standard deviation is the square root of the sum of the squared standard deviations of the two distributions  $\mathcal{N}\left(0 + 0 = \mathbf{0},\left(\sigma_1^2+\sigma_2^2\right) \mathbf{I}\right)$  (see [[Gaussian Distribution#Merging two Gaussian Distributions]]). The new mean will be $0 + 0 = 0$ and the new standard deviation will be $\sqrt{\left(1-\alpha_t\right)+\alpha_t\left(1-\alpha_{t-1}\right)}=\sqrt{1-\alpha_t \alpha_{t-1}}$. We can therefore write this as:
$$x_t = \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2}+\sqrt{1-\alpha_t \alpha_{t-1}} \bar{\epsilon}_{t-2}$$
where $\overline{\boldsymbol{\epsilon}}_{t-2}$ merges two Gaussians.

This can be extended to $t - 3$ to $t$ as well:
$$\begin{aligned}
&x_t=\sqrt{\alpha_t \alpha_{t-1} \alpha_{t-2}} x_{t-3}+\sqrt{1-\alpha_t \alpha_{t-1} \alpha_{t-2}} \epsilon
\end{aligned}$$
And then to an arbitrary $t$:
$$x_t =\sqrt{\alpha_t \alpha_{t-1} \ldots \alpha_1 \alpha_0} x_0+\sqrt{1-\alpha_t \alpha_{t-1} \ldots \alpha_1 \alpha_0} \epsilon$$
We can then define a new scalar $\bar{\alpha}_t=\prod_{s=1}^t \alpha_s$ which is the product of the $\alpha_t$ at each step and then end up with a function that lets us directly sample time step $t$ starting with time step $0$ without needing to compute the intermediate steps.

The new distribution is given by:
$$\left.q\left(\mathbf{x}_t \mid \mathbf{x}_0\right)=\mathcal{N}\left(\mathbf{x}_t ; \sqrt{\bar{\alpha}_t} \mathbf{x}_0,\left(1-\bar{\alpha}_t\right) \mathbf{I}\right)\right)$$
And we can sample from it using the reparameterization trick:
$$q(x_t|x_0) =\sqrt{\bar{\alpha}_t} x_0+\sqrt{1-\bar{\alpha}_t} \epsilon$$
See [here](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) for more details.

> [!NOTE] We can go from time 0 to time $t$ without intermediate steps using:
> $$q(x_t|x_0) =\sqrt{\bar{\alpha}_t} x_0+\sqrt{1-\bar{\alpha}_t} \epsilon$$
