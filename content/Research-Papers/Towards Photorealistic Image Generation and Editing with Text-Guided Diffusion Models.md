---
tags: [flashcards]
aliases: [GLIDE]
source: https://arxiv.org/abs/2112.10741
summary:
---

This mode was a precursor to DALL-Ev2. GLIDE produced images that were preferred by human annotators compared to DALL-Ev1.

[[Diffusion Models Beat GANs on Image Synthesis]] used [[Diffusion Models Beat GANs on Image Synthesis#Classifier Guidance|Classifier Guidance]] to improve image quality with lower diversity of samples by conditioning on a class label with an additional image classifier.

GLIDE can condition on textual captions. It can generate images and do image in-painting. GLIDE stands for **G**uided **L**anguage to **I**mage **D**iffusion for **G**eneration and **E**diting.

GLIDE uses [[Classifier-Free Diffusion Guidance]]. It adds the ability to condition the model on generic text prompts by sometimes replacing text captions with an empty sequence (referred to as $\emptyset$) during training. The model is then guided towards the caption $c$ using the modified prediction $\hat{\epsilon}$:
$$\hat{\epsilon}_\theta\left(x_t \mid c\right)=\epsilon_\theta\left(x_t \mid \emptyset\right)+s \cdot\left(\epsilon_\theta\left(x_t \mid c\right)-\epsilon_\theta\left(x_t \mid \emptyset\right)\right)$$

It also tried to use [[CLIP Learning Transferable Visual Models From Natural Language Supervision|CLIP]] guidance with text prompts.
https://youtu.be/c1GwVg3lt1c?t=450