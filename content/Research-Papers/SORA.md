---
tags:
  - flashcards
source: https://openai.com/sora
summary: A diffusion model from OpenAI that can generate extremely high-quality videos from a text prompt.
---
- Sora is a [[AI-Notes/Concepts/Diffusion Models/Diffusion Models|Diffusion Model]], which generates a video by starting off with one that looks like static noise and gradually transforms it by removing the noise over many steps.
- Similar to GPT models, Sora uses a [[AI-Notes/Transformers/Transformer|Transformer]] architecture, unlocking superior scaling performance.
- It uses the recaptioning technique from DALL·E 3, which involves generating highly descriptive captions for the visual training data.
- In addition to being able to generate a video solely from text instructions, the model is able to take an existing still image and generate a video from it, animating the image’s contents with accuracy and attention to small detail.
- They plan to include [[AI-Notes/Definitions/C2PA Metadata|C2PA Metadata]] in the videos which will identify content as being created by AI.