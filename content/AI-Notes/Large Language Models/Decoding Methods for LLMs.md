---
tags:
  - flashcards
source: https://huggingface.co/blog/how-to-generate
summary: Different methods for generating sequences of text from LLMs
publish: true
---
All of the following decoding methods can be used for auto-regressive language models.

> [!NOTE] Auto-regressive language generation
> Auto-regressive language generation is based on the assumption that the probability distribution of a word sequence can be decomposed into the product of conditional next word distributions. 

### Greedy Search
- Greedy search is the simplest decoding method. It selects the word with the highest probability as its next word.

### Beam Search
- Beam search reduces the risk of missing hidden high probability word sequences by keeping the most likely num_beams of hypotheses at each time step and eventually choosing the hypothesis that has the overall highest probability.

### Temperature Scaling
- A trick to make the distribution P(w|w1:t−1) sharper (increasing the likelihood of high probability words and decreasing the likelihood of low probability words) is by lowering the so-called temperature of the softmax.
- Setting temperature → 0, temperature-scaled sampling becomes equal to greedy decoding.

For more details see [[AI-Notes/Activation/Softmax#Softmax with temperature scaling|Softmax]].

### Top-K Sampling
- In Top-K sampling, the K most likely next words are filtered and the probability mass is redistributed among only those K next words.
- In its most basic form, sampling means randomly picking the next word w_t according to its conditional probability distribution.

### Top-P Sampling
- Limiting the sample pool to a fixed size K could endanger the model to produce gibberish for sharp distributions and limit the model's creativity for flat distributions. This intuition led Ari Holtzman et al. (2019) to create Top-p or nucleus-sampling.
- In Top-p sampling, the method chooses from the smallest possible set of words whose cumulative probability exceeds the probability p.
- Top-p can also be used in combination with Top-K, which can avoid very low ranked words while allowing for some dynamic selection.
- As ad-hoc decoding methods, top-p and top-K sampling seem to produce more fluent text than traditional greedy and beam search on open-ended language generation.