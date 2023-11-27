---
tags: [flashcards]
source: https://vaclavkosar.com/ml/cross-attention-in-transformer-architecture
summary: cross-attention combines two seperate embedding sequences of the same dimension (one serves as query and one as key/value inputs)
---

Cross-attention combines asymmetrically ==two separate embedding sequences of same dimension==, in contrast self-attention input is a single embedding sequence. One of the sequences serves as a query input, while the other as a key and value inputs.
<!--SR:!2023-12-11,119,270-->

Cross attention is:
- an [attention mechanism in Transformer architecture](https://vaclavkosar.com/ml/transformers-self-attention-mechanism-simplified) that mixes two different embedding sequences
- the two sequences must have the same dimension
- the two sequences can be of different modalities (e.g. text, image, sound)
- one of the sequences defines the output length as it plays a role of a query input
- the other sequence then produces key and value input

Cross attention is a way to condition a model with some additional information. It is used in many SOTA models like [[Latent Diffusion Models|Stable Diffusion Models]], ImageGen, Muse, etc. to condition models on text input. It can be added into existing models because it can take an input, mix in an extra source of information, and then provide an output with the same dimensions as the input.