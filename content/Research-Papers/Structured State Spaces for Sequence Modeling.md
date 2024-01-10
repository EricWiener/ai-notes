---
tags:
  - flashcards
source: 
summary: 
aliases:
  - S4
---
[Stanford Blog Post 4 Part Series](https://hazyresearch.stanford.edu/blog/2022-01-14-s4-1)

[Stanford MLSys YouTube Talk](https://youtu.be/EvQ3ncuriCM)

[MatLab YouTube Video on State-Space Equations](https://youtu.be/hpeKrMG-WP0)

[Another YouTube Video on Structured State Space Models for Deep Sequence Modeling](https://youtu.be/OpJMn8T7Z34)

See [[Research-Papers/structured-state-spaces-for-sequence-modeling-srcs/The State Space Model|The State Space Model]].


> [!NOTE] Structured State Spaces (S4)
> This is a new sequence model based on [[Research-Papers/structured-state-spaces-for-sequence-modeling-srcs/The State Space Model|The State Space Model]] that is "**continuous-time** in nature, excels at **modeling long dependencies**, and is very **computationally efficient**."


**Definitions:**
-  [[AI-Notes/Definitions/Ordinary Differential Equation|ODE]]
# Continuous time series
The S4 model was created to handle the class of data they refer to as "continuous time series." This is data that consists of very long, smoothly varying data. It typically comes from sampling an underlying continuous-time process (ex. you discretize a continuous signal to get a digital representation of audio, biometric signals, video, measurement, etc.).

![[Research-Papers/structured-state-spaces-for-sequence-modeling-srcs/structured-state-spaces-for-sequence-modeling-20231229085801595.png]]
["Continuous time series" are characterized by very long sequences sampled from an underlying continuous process](https://hazyresearch.stanford.edu/blog/2022-01-14-s4-1)

**Desirable properties when handling continuous data:**
- Handle information across long distances (even short segments of speech data contain thousands of steps).
- Not be sensitive to the resolution of the data (the model should work regardless of whether the signal was sampled at 100Hz or 200Hz).
- Be efficient at training and inference.

# Existing Approaches.
![[AI-Notes/Attention/Attention#Processing Sequences]]

### Continuous Time