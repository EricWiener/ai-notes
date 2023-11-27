---
tags: [flashcards]
source:
summary: improves on [[End-to-End Object Detection with Transformers|DETR]] by  formulating queries in decoder as dynamic anchor boxes and refine them step-by-step across decoder layers.
---

Changes the learnable object queries in the DETR Transformer Decoder to be 4D anchor boxes (x, y, width, and height). It also updates these anchor boxes in each layer of the decoder (vs. DETR which passes the same learned object queries to each decoder layer during a single training step). These changes help the model converge faster, improve performance, and increase interpretability.

# Queries in [[DETR]]
DETR uses 100 learnable queries to probe and pool features from images. It then makes predictions without the need for non-max suppression. This paper believes that the query design is inefficient and one of the causes of the slow training. Additionally, the role of the learned queries is not fully understood.

![[detr-cross-attention-layer-annotated]]
The above figure shows the cross-attention part of DETR's Transformer decoder. The same learnable queries fed to all the layers of the decoder during a single training step. The paper argues this accounts for its slow training convergence.

The queries in DETR are formed by two parts: 
- A content part (the decoder self-attention output denoted "Decoder Embeddings" in the diagram above)
- A positional part ("Learnable Queries")

The keys consist of two parts as well:
- A content part (encoded image features)
- A positional part (positional encodings)

The cross-attention weights are computed by comparing a query with a set of keys. Therefore, queries can be interpreted as pooling features from a feature map based on the query-to-feature similarity measure which considers both content and positional information. The content similarity is for pooling ==semantically related features==, the positional similarity is to provide a ==positional constraint for pooling features around the query position==.
<!--SR:!2024-09-02,414,272!2024-07-13,399,310-->

# Queries in DAB-DETR
![[dab-detr-queries.png|400]]
The queries in DAB-DETR are 4D anchor points and they are updated after every layer of the decoder. This makes them dynamic.

By modifying the learnable queries to be anchor boxes, they change the positional information contained in the queries to the decoder. They do not directly modify the content information in the queries (the decoder embeddings) or the positional/content information in the keys.

**Why can you not do layer-by-layer query refinement in traditional [[DETR]] without using 4D anchor points?**
??
Because the original queries are high-dimensional embeddings so it is unclear how to take a high-dimensional embedding, convert it to a bounding box, update it, and then convert it back to the original embedding space.
<!--SR:!2025-02-10,575,330-->

> [!QUESTION] What does modulating cross-attention map mean?
> This motivates using the center position (x, y) of an anchor box to pool features around the center and use the anchor box size (w, h) to modulate the cross-attention map, adapting it to anchor box size.


The paper's reasoning for the queries being an improvement is:
> These queries add better spatial priors for the cross-attention module by considering both the position and size of each anchor box. This also simplies the implementation and gives a deeper understanding of the role of queries in DETR.


> [!QUESTION] I'm a little iffy on what "better spatial priors" means.


> [!QUESTION] Title
> Using box coordinates not only helps using explicit positional priors to improve the query- to-feature similarity and eliminate the slow training convergence issue in DETR, but also allows us to modulate the positional attention map using the box width and height information.



# Related Work

> [!NOTE] The general approach of related work is to make each query in DETR more explicitly associated with one specific spatial position rather than multiple positions.

Conditional DETR learns a conditional spatial query by adapting a query based on its content feature for better matching with image features (Meng et al., 2021). Efficient DETR introduces a dense prediction module to select top-K object queries (Yao et al., 2021) and Anchor DETR formulates queries as 2D anchor points (Wang et al., 2021), both associating each query with a specific spatial position. Similarly, [[Deformable DETR]] directly treats 2D reference points as queries and performs [[Deformable Attention|deformable cross-attention]] operation at each reference points (Zhu et al., 2021). But all the above works only leverage 2D positions as anchor points without considering of the object scales.

### What's the difference between DAB-DETR and [[Deformable DETR]] (D-DETR)?
![[detr-vs-deformable-detr-vs-dab-detr.png]]
Deformable DETR uses high-dimensional queries to pass positional information to each layer. These queries don't have a clear semantic meaning and are not updated layer by layer (hard to predict anchors, update them, and then return to original embedding space). DAB-DETR, on the other hand, replaces the learnable queries with 4D anchor boxes.

Note that both Deformable DETR and DAB-DETR will update the bounding box predictions at each stage of the decoder.