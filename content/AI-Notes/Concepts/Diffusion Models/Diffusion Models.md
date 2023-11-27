---
tags: [flashcards]
source: https://scale.com/guides/diffusion-models-guide
summary: diffusion models can generate extremely high quality and diverse images from noise. They are trained by corrupting training data and then learning to recover the original image.
---

![[diffusion-model-generated-images.png]]
### What are Diffusion Models?
Generative models are a class of machine learning models that can generate new data based on training data. Other generative models include [[Generative Adversarial Networks]] (GANs),  [[Autoencoders#Variational Autoencoders|Variational Autoencoders]] (VAEs), and Flow-based models.

At a high level, Diffusion models work by destroying training data by adding noise using a fixed algorithm and then learn to recover the data by reversing this noising process (this is learned). In other words, Diffusion models can generate coherent images from noise. 

Diffusion models train by adding noise to images, which the model then learns how to remove. The model then applies this denoising process to random seeds to generate realistic images.

# Papers
- Dalle (image gen from text)
- Dalle 2 (image gen from text)
- Diffusion models now being applied to video generation.  In [this](https://plai.cs.ubc.ca/2022/05/20/flexible-diffusion-modeling-of-long-videos/) paper, they generate 1+ hour videos that are consistent!
-  [eDiffi: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers](https://arxiv.org/pdf/2211.01324.pdf) which is NVIDIA’s most recent contribution to the text-to-image diffusion landscape.
- This is a really nice tutorial on how to build your own minimal Imagen ([https://imagen.research.google/](https://imagen.research.google/)...[https://www.assemblyai.com/blog/build-your-own-imagen-text-to-image-model/](https://www.assemblyai.com/blog/build-your-own-imagen-text-to-image-model/).  Covers u-nets, diffusion models, sampling approaches. Imagen is a text to image framework from Google.
- Questioning the outputs of diffusion models like DALLE and Stable Diffusion: [Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models](https://arxiv.org/abs/2212.03860).   This is what the authors have tried to question and investigate: ‘.._Do_ _diffusion_ _models create unique works of art, or are they stealing content directly from their training sets?_’
- At tomorrow’s paper party I’ll be presenting [Elucidating the Design Space of Diffusion-Based Generative Models](http://arxiv.org/abs/2206.00364). Some general familiarity with diffusion will definitely help. The beginning part of [this CVPR tutorial](https://www.youtube.com/watch?v=cS6JQpEY9cs) is a good resource.
- DiffusionInst: Diffusion Model for Instance Segmentation. [Arvix](https://ar5iv.labs.arxiv.org/html/2212.02773)
- DiffusionDet: Diffusion Model for Object Detection. [GitHub](https://github.com/ShoufaChen/DiffusionDet)
- Diffusion Models for Video Prediction and Infilling. [Arvix](https://ar5iv.labs.arxiv.org/html/2206.07696). [Website](https://sites.google.com/view/video-diffusion-prediction)
- RaMViD - Random-Mask Video Diffusion. [GitHub](https://github.com/Tobi-r9/RaMViD)
- Flexible Diffusion Modeling of Long Videos. [Website](https://plai.cs.ubc.ca/2022/05/20/flexible-diffusion-modeling-of-long-videos/)
- "Diffusion Models Beat GANs on Image Synthesis" Dhariwal & Nichol, **OpenAl**, 2021. Shows power of diffusion models.
- "Cascaded Diffusion Models for High Fidelity Image Generation". Ho et al., **Google**, 2021. Shows power of diffusion models.
Normalizing Flows: the whole thing has to be differentiable. In Diffusion models, at inference time you call the model multiple times (for each step of denoising) and you can do non-differentiable stuff.

### Text-to-Image Generation
- DALL-E, DALL-E 2 (OpenAI)
- Imagen (Google)

# Zoox Presentation
You can view diffusion either as a discrete markov problem or as a differential equation.

### Markov Interpretation

![[screenshot-2023-03-12_14-07-17.png]]
You first use a fixed algorithm to add noise (in this example you add Gaussian noise). You keep adding noise until you get random noise. You then use the model to try to get rid of the noise you added.

$x_0$ is your original data. You then have conditional distributions $x_1 | x_0$ ($x_1$ conditioned on $x_0$), $x_2 | x_1$, etc. which describe the noise adding process. Because we are adding Gaussian noise, the distribution of $x_1 | x_0$ is going to be some Gaussian distribution.

During inference you assume $x_T$ is an approximately uniform Gaussian and you ask the model to do the inverse noising process to generate $x_0$ by predicting $x_{T - 1}, x_{T-2}, \ldots, x_0$ one at a time until you get back to $x_0$.

You can also think of difussion as a [[Autoencoders]] where the encoder is fixed and the decoder is learned.


### Differential Equation Interpretation
![[screenshot-2023-03-12_14-22-47.png]]
You can think of diffusion as a differential equation if you have your time steps be very small $\epsilon$. You then integrate over your function that adds noise to go from the original image to the noised image.

Thinking of the problem as a differential equation problem lets you benefit from papers that look at optimizing differential equations.

Having a continious distribution makes it easier to view the problem than having it be discrete.