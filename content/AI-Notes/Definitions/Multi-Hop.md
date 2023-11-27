---
tags: [flashcards]
source:
summary: multi-hop means that a computational step is performed multiple times on an input before generating an output (ex. needing to look at multiple sentences in a paragraph in order to answer a question or track something through multiple frames of a video)
---

[[Non-Local Neural Networks]]: multiple non-local blocks can perform long-range multi-hop communication. Messages can be delivered back and forth between distant positions in spacetime, which is hard to do via local models.
[[Non-Local Neural Networks]]: multi-hop dependency modeling, e.g., when messages need to be delivered back and forth between distant positions, is difficult for traditional and recurrent networks.

We propose a multi-hop attention for the Transformer. It refines the attention for an output symbol by integrating that of each head, and consists of two hops. The first hop attention is the scaled dot-product attention which is the same attention mechanism used in the original Transformer. The second hop attention is a combination of multi-layer perceptron (MLP) attention and head gate, which ef- ficiently increases the complexity of the model by adding dependencies between heads. (Attention over Heads: A Multi-Hop Attention for Neural Machine Translation.

Multi-hop attention was first proposed in end-to-end memory networks (Sukhbaatar et al., 2015) for machine comprehension. In this paper, we define a hop as a computational step which could be performed for an output symbol many times. By “multi-hop attention”, we mean that some kind of attention is calculated many times for generating an output symbol.  (Attention over Heads: A Multi-Hop Attention for Neural Machine Translation)

Words have one-hop relationship with entities are called primary triggers, and words have two-hop relationship with primary triggers are called secondary triggers. (Low-Resource Named Entity Recognition Based on Multi-hop Dependency Trigger)

Multi-hop MRC datasets require a model to read and perform multi-hop reasoning over multiple paragraphs to answer a question. (Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps)

![[screenshot-2022-03-10_08-04-40.png]]
- multiple computational steps (hops) are performed per output symbol
- multiple computational steps (which we term “hops”) per output symbol. We will show experimentally that the multiple hops over the long-term memory are crucial to good performance of our model on these tasks
- [[End-To-End Memory Networks, Sainbayar Sukhbaatar et al., 2015.pdf]]


 transforms images gradually between two domains, through multiple hops. Instead of executing translation directly, we steer the translation by requiring the network to produce in-between images that resemble weighted hybrids between images from the input domains. [GANHopper: Multi-Hop GAN for Unsupervised Image-to-Image Translation](https://arxiv.org/abs/2002.10102)


      
![[screenshot-2022-03-10_08-15-37.png]]
As an intuitive example, consider the dialogue in Fig. 1 through which one speaker localizes the second girl in the image, the one who does not “have a blue frisbee.” For this task, a single-hop model must determine upfront what steps of reasoning to carry out over the image and in what order; thus, it might decide in a single shot to highlight feature maps throughout the visual network detecting either non-blue colors or girls. In contrast, a multi-hop model may first determine the most immediate step of reasoning necessary (i.e., locate the girls), highlight the relevant visual features, and then determine the next immediate step of reasoning necessary (i.e., locate the blue frisbee), and so on. While it may be appropriate to reason in either way, the latter approach may scale better to **longer language inputs and/or or to ambiguous images where the full sequence of reasoning steps is hard to determine upfront**, which can even be further enhanced by having intermediate feedback while processing the image.

In this paper, we therefore explore several approaches to generating FiLM parameters in multiple hops. These approaches introduce an intermediate con- text embedding that controls the language and visual processing, and they al- ternate between updating the context embedding via an attention mechanism over the language sequence (and optionally by incorporating image activations) and predicting the FiLM parameters. (Visual Reasoning with Multi-hop Feature Modulation)