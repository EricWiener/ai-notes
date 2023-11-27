---
aliases: [ConvS2S]
---

# Abstract
- [[Voice Conversion]] (VC) method using sequence-to-sequence (seq2seq or S2S) learning
    - In voice conversion, we **change the speaker identity from one to another, while keeping the linguistic content unchanged**.
- This paper uses a fully convolitional architecture since it enables parallel computation (vs. an RNN). Can also use normalization techniques like [[Research Papers/Batch Normalization]].

# Introduction
- A typical ([[VC]]) pipeline of the training process consists of extracting acoustic features from source and target utterances, performing dynamic time warping (DTW) to obtain time-aligned parallel data, and training an acoustic model that maps the source features to the target features frame-by-frame.
- This paper modifies the existing ConvS2S architecture to work for vision.
- The original ConvS2S architecture used a decoder with [[Casual Convolutions]]