[[Language Modeling with Gated Convolutional Networks, Yann N. Dauphin et al., 2016 1.pdf]]

# Key Take-Aways
- This paper proposes a gating mechanism (similar to LSTMs) for convolutional layers.
- This paper applies the gated convolutions to language modeling.

![[architecture-gated-convolutional-network-language-model.png]]

# Abstract
- pre-dominant approach to language mod- eling to date is based on recurrent neural net- works
- stacked convolutions, which can be more efficient since they allow parallelization over sequential tokens.

# Introduction
- Convolutional networks can be stacked to represent large context sizes and extract hierarchical features over larger and larger contexts with more abstractive features.
    - This allows them to model long-term dependen- cies by applying $O(\frac{N}{K})$ operations over a context of size $N$ and kernel width $k$.
    - Recurrent networks view the input as a chain structure and therefore require a linear number $O(N)$ of operations.
- The computational of all inputs words can be performed simultaneously using conv layers.
- Our gated linear units reduce the vanishing gradient problem for deep architectures by providing a linear path for the gradients while retaining non-linear capabilities.
    - higher accuracy and converge faster than the LSTM-style gating
    
# Approach
- Approach has no temporal dependencies and is easier to parallelize over the individual words of a sentence.
- Compared to recurrent networks, the context size is finite
