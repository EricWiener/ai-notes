---
tags: [flashcards]
source: https://youtu.be/HoKDTa5jHvg
summary:
---

# General Idea
We apply a lot of noise to an image and then train a model to reverse this process. We can then start with random noise and have the model remove the noise until we end up with a new image.

### Forward Diffusion Process
This is applying noise to an image in an iterative fashion. The image will eventually become pure noise. The noise is sampled from a normal distribution.

The amount of noise added varies at each time step. Latter steps will add more noise to the image. The original 2015 paper used a linear schedule and latter papers by OpenAI switched to using a cosine schedule.

### Reverse Diffusion Process
This is going from just noise to an image. The model gradually removes the noise until you end up with an image from a similar distribution as the training data.

> [!NOTE] Why remove noise iteratively vs. in one shot?
> You remove noise incrementally (vs. just one step) because the original 2015 paper said a one-shot approach results in worse outcomes.

You have three options for the model to predict: the original image, the mean of the noise to subtract, and the noise to subtract from the current image to get the next image. The first option doesn't work for the above reason and the latter two options end up being the same, but the papers all decided to predict the noise to subtract.

> [!NOTE] Why is the mean of the noise and the noise of the image the same?
> A normal distribution needs both mean and variance so it would seem you need to predict both the mean and variance in order to figure out the noise. In the original 2015 paper the variance was fixed so this did not need to be predicted. However, latter papers did also make the variance a learned parameter.

### Model
The model used by the 2020 paper was a U-Net with attention blocks and skip connections. You use the same model at each time step. You tell the model what time step you are at using a sinusoidal embedding that is projected into each residual block. Since the forward diffusion process adds different amounts of noise at different time steps, you want the model to be able to remove different amounts of noise depending on the time step.

# Math
We define the following:
- $x_t$ is the image at time step $t$. Ex: $x_0$ is the original image. $x_T$ is the final noised image.
-  $q(x_t|x_{t-1})$ is the forward process. It takes in an image $x_{t_1}$ and returns an image $x_t$ with a little more noise added.
- $p(x_{t-1}|x_t)$ is the reverse process. It takes an image $x_t$ and produces a sample with slightly less noise $x_{t-1}$ using the neural network.

**Forward process $q(x_t|x_{t-1})$:**
$$\mathrm{q}\left(\mathrm{x}_t \mid x_{t-1}\right)=\mathcal{N}\left(x_t, \sqrt{1-\beta_t} x_{t-1}, \beta_t I\right)$$


# Paper Overview
