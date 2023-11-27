---
tags: [flashcards]
aliases: [High-Resolution Image Synthesis with Latent Diffusion Models, Stable Diffusion Models]
source: https://youtu.be/J87hffSMB60
summary:
---

> [!NOTE]
> Stable Diffusion is a type of Latent Diffusion Model that was sponsored by stability.ai and trained on an aesthetic dataset that made it good for text-to-image generation.

Diffusion Models use U-Net architectures where the output of the model has the same spatial resolution as the input. This makes it very expensive to work on high resolution images (ex. 1024x1024) because your input and output are very large. Additionally, when sampling new images from noise, you need to repeat the noise-prediction for $t$ steps where $t$ can be something like 150.

### [[Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models]] approach to large spatial resolution
Train the model on smaller images (ex. 256 x 256) and then train another network to upsample the de-noised images.

At inference time you downsample your inputs to the model and upsample the outputs of the model.
![[glide-approach-to-large-spatial-resolutions.png]]

### LDM Approach:
Latent Diffusion Models work in the latent space (the compressed data space) instead of the image space. They use an encoder-decoder to work with a lower-dimensionality representation of the image vs. the original image (replacing the downsampling/upsampling used by [[Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models]]).

![[latent-diffusion-model-architecture.png]]

You first train the encoder-decoder on its own:
![[encoder-decoder-training.png]]

When training your diffusion model, you apply noise onto the lower dimensional representation of the data (not the original image) and then you pass the de-noised image to the decoder:
![[ldm-noise-adding.png]]

### LDM Benefits
Can run much faster and with less memory.

### Stable Diffusion Models
Stable Diffusion is a Latent Diffusion Model that was trained on an aesthetic dataset that was created using a [[CLIP Learning Transferable Visual Models From Natural Language Supervision|CLIP]]-based model that filtered another dataset based on how beautiful an image was. It was made after the initial Latent Diffusion Model paper and was a collaboration between the original authors and stability.ai.

This training data makes the model have its own style that is different from other text-to-image generators.