---
tags: [flashcards, eecs498-dl4cv]
source: 
summary:
---

**Computer Vision:** Building artificial systems that process, perceive, and reason about visual data

**Learning:** Building artificial systems that learn from data and experience

**Deep Learning:** Hierarchical learning algorithms with many “layers”, (very) loosely inspired by the brain
- Involves a lot of non-linearity

**Artificial intelligence:** building systems with intelligent behavior. Can do non-trivial stuff that feels smart to humans.
![[diagram-of-what-498-covers.png]]
What EECS 498 DL4CV covers.

# History
**Hubel and Wiesel, 1959**
They were able to measure individual neurons in a cats brain when they changed what the cat saw. Were hoping to show the cat different images and see what neurons fired. What turned out is that the patterns were very complicated. Weren't very clear signals for why the neurons were firing.

Were only able to tell what happened by looking at the neurons firing when they changed slides (they had an old slide projector so there was a dark bar that moved across the screen when the slide changed). There are some simple cells in the brain that respond to light orientation. They feed up into complex cells that respond to general orientations and movement. Hypercomplex cells can recognize even more complex cells.

> [!NOTE]
> Hubel and Wiesel developed a hierarchical model of the visual pathway with neurons in lower areas such as V1 responding to features such as oriented edges and bars, and in higher areas to more specific stimuli


**The Summer Vision Project**
Report from 1966. People at MIT wanted to get together researchers and just finish computer vision as a field.

**David Mar**
Put forward a hypothesized pipeline:
- input image
- edge image
- 2 1/2 D sketch
- 3-D model

**Recognition via Edge Detection**
Canny Edge Detection was developed in 1980s.

**SIFT: Recognition via Matchings (2000)**
![[CleanShot_2022-01-01_at_13.04.202x.png]]

Matching objects using similarly known objects

**Hubel and Wiesel's** **Perceptron**
One of the earliest algorithms that could learn from data (1957). It was implemented in hardware. Recognize it as a linear classifier. Could learn to recognize letters of the alphabet.

**Minsky and Papert (1969)**
Pointed out Perceptron has a certain mathematical structure and wasn't magic. Pointed out it could never learn the XOR function. It doused all the hype about the perceptron learning algorithm. 

**Neocognitron**
Neural network architecture for pattern recognition inspired by Hubel and Wiesel's Perceptron. Looked like convolution and pooling that we use today. Looks very similar to AlexNet.

**LeCun et al.'s Neural Computation**
Used backpropagation to train the weights of the Neocognitron.

**Hardware has sped up computation and made it cheaper**
![[CleanShot_2022-01-01_at_13.09.312x.png]]

![[CleanShot_2022-01-01_at_13.11.022x.png]]

The bottom is the above graph squashed down. Tensor Cores have significantly increased GFLOPs per dollar.

**When is deep learning not superior to traditional?**
- Currently traditional methods are used for 3D computer vision (like scene reconstruction).
- When you don’t have a lot of data