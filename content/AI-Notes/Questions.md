

### Motion Prediction
For joint motion forecasting there are a lot of papers depending on what specifically you’re looking for  
-   [http://arxiv.org/abs/2106.08417](http://arxiv.org/abs/2106.08417)
-   [http://arxiv.org/abs/2203.01880](http://arxiv.org/abs/2203.01880)
-   [http://arxiv.org/abs/2111.13566](http://arxiv.org/abs/2111.13566)
-   [https://arxiv.org/abs/2111.14973](https://arxiv.org/abs/2111.14973)
-   [http://arxiv.org/abs/1911.00997](http://arxiv.org/abs/1911.00997)
-   [https://arxiv.org/abs/2110.06607v3](https://arxiv.org/abs/2110.06607v3)

### Tomorrow's work:
- [x] [[DETR|DETR]]
    - [ ] What's the difference between attention and [[Non-Local Neural Networks]]
- [ ] [[Reversible Vision Transformers]] 
- [ ] Zero-optimizer (Zero: Memory optimizations toward training trillion parameter models)
- [ ] [[SimMIM]]
- [x] TransFusion
- [ ] What is self-learning (mentioned in [[Developing and Deploying Conversational AI at Scale]])
- [ ] What is a Markov Chain?
- [ ] Focal Modulation Networks (recommended in zoox machine learning): https://arxiv.org/abs/2203.11926

- [ ] Lecture series that gives good overview of explainable ai basics:
did a good job at giving an overview of various explainable ai basics  
[https://m.youtube.com/playlist?list=PLoROMvodv4rPh6wa6PGcHH6vMG9sEIPxL](https://m.youtube.com/playlist?list=PLoROMvodv4rPh6wa6PGcHH6vMG9sEIPxL)  
[https://docs.google.com/presentation/d/1khY_li29A5aUo_cEVRsvO8pcRn7Xp9Bi/mobilepresent?slide=id.p1](https://docs.google.com/presentation/d/1khY_li29A5aUo_cEVRsvO8pcRn7Xp9Bi/mobilepresent?slide=id.p1)

### Backlog
- [x] [[DenseNet]]
- [ ] How do [[Deep Roots Improving CNN Efficiency with Hierarchical Filter Groups|Filter Groups]] have any benefits? Also add to flashcards
- [ ] Layernorm and Groupnorm notes
- [ ] What are graph neural networks
- [ ] https://blogs.nvidia.com/blog/2022/03/23/what-is-path-tracing/
- [ ] https://blogs.nvidia.com/blog/2022/03/22/h100-transformer-engine/
- [ ] [[NeRF]]

### Paper Backlog:
- [ ] [Seq2seq and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html)
- [ ] [Deep Learning: The Transformer. Sequence-to-Sequence (Seq2Seq) models… | by Mohammed Terry-Jack | Medium](https://medium.com/@b.terryjack/deep-learning-the-transformer-9ae5e9c5a190)
- [ ] SA-Det3D: Self-Attention Based Context-Aware 3D Object Detection (self-attention for 3D)
- [ ] VectorNet: https://blog.waymo.com/2020/05/vectornet.html
- [ ] Gated CNN: Integrating multi-scale feature layers for object detection
- [ ]  Transfer learning papers (recommended by Allan from Zoox):
    - [ ]  [http://taskonomy.stanford.edu/](http://taskonomy.stanford.edu/)
    - [ ]  [https://arxiv.org/abs/2006.04096](https://arxiv.org/abs/2006.04096)
- [ ]  Training with large batch sizes: Layer-wise Adaptive Learning Rates (LARS)
- [ ]  Deep Fried Convnets (left off mid-way)
- [ ]  Network in Network (barely scratched surface - this introduces global average pooling)
- [ ]  Going Deeper with Convolutions (GoogleNet paper)
- [ ] CNN-like architectures with self-attention:  (Wang et al., 2018; Carion et al., 2020),
- [ ] Vision w/ attention replacing the convolutions entirely (Ramachandran et al., 2019; Wang et al., 2020a) 

### Question Dump:
- [ ] How to use attention with CNN's?
- [ ] For [[Optimized Autoregressive Transformer]] how does saving the hidden states of the decoder speed up computation?
- [ ] What are EfficientNet's

[Good article on different conv types](https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215)

- **MobileNet Questions**
    - [x]  Flattened networks build a network out of fully factorized convolutions (Flattened convolutional neural networks for feedforward acceleration)
    - [ ]  Factorized Networks introduces factorized convolutions + topological connections (Factorized convolutional neural networks)
    - [ ]  Distillation uses larger networks to teach a smaller network (Distilling the knowledge in a neural network)
    - [ ]  Transform networks (28)
    - [ ]  Deep fried convnets (37)
        - [ ]  What is meant by “increase model discriminability for local patches within the receptive field”
        - [ ]  What is a rank-one assumption?
- Nonlocal block
    - [ ]  BottleNeck block
    - [ ]  Inflated variant of convolutional networks (7) → going from 2D to 3D
    - [ ]  Multi-scale testing
    - [ ]  Conditional random fields (CRF - 29, 28). In the context of deep neural networks, a CRF can be exploited to post-process semantic segmentation predictions of a network
    - [ ]  *Interaction Networks* (IN) [2, 52] were proposed recently for modeling physical systems. They operate on graphs of objects involved in pairwise interactions. Hoshen [24] presented the more efficient Vertex Attention IN (VAIN) in the context of multi-agent predictive modeling
- GroupNorm
- Transformers: BERT, GPT, GPT-2, GPT-3
- Forward vs reverse node differentiation:
[https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/autograd/](https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/autograd/)
- [ ]  If you prune a network, how does it run faster (aren’t you doing matrix multiplies? Or is the underlying structure just 1D vectors)?
- [ ] The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks, Frankle and Carbin, 2019
- [ ] Fractional Max Pooling. See [here]]

- Look into “Modular Networks” where one network predicts what program to use

# Done
- [x] Cross Stage Partial Network
- [x] this paper saying replace attention layer with average pooling layer can also achieve pretty good performance in transformer [ARVIX](https://arxiv.org/abs/2111.11418)
- [x] Read and understand [[ViT An Image is Worth 16x16 Words Transformers for Image Recognition at Scale|ViT]]
- [x] How do embeddings work?
- [x] What's homoscedastic loss? [[Homoscedastic loss]]
- [x] What's beam search?
    - [[Beam Search]]
- [x] What's the benefit of multiple attention heads? [[Attention#Multihead Self-Attention Layer]]
- [x] How do [[Positional Encoding]] work? (like in BeRT and ViT)
- [x]  Self-attention (49) [[Attention#Self-Attention Layer]]
- [x] Multi-hop dependency modeling (eg. when messages need to be delivered back and forth between distant positions)
    - [x] https://www.reddit.com/r/MLQuestions/comments/bqhurs/seen_papers_that_list_multihop_dependency/
    - [[Multi-Hop]]
- [x] What are Deep Sets?
- [x] Review [[Non-Local Neural Networks]]
- [x] [[GRU]]
- [x] How does numpy einsum work? [Possibly useful SO post](https://stackoverflow.com/questions/26089893/understanding-numpys-einsum)
- What are asynchronous solvers (ASGD)? In context of training large scale distributed networks
- What is asynchronous gradient descent?
- [x] LayerScale
- [x] Swin v2
- [x] Finish 498 VIT lecture