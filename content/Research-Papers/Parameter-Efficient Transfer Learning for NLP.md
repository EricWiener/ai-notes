---
tags:
  - flashcards
source: https://arxiv.org/abs/1902.00751
summary: 
aliases:
  - Adapter Modules
---
This paper introduced using Adapter Modules for transfer learning for large language models. These are new layers that are added throughout the module and are trained for a specific task. You then use a different set of weights for these additional layers depending on the task.
### Adapters
Adapters are related to [[Parameter Efficient Fine-Tuning|Prefix Tuning]] in that they also add additional parameters to each transformer block. However, instead of adding prefixes to the input embeddings, the adapter adds adapter layers:
![[understanding-parameter-efficient-finetuning-of-large-language-models-20231009120208697.png]]
![[understanding-parameter-efficient-finetuning-of-large-language-models-20231009120214657.png|300]]

Adapter layers are usually small and have a bottleneck structure similar to [[Autoencoders]] where the first layers reduces the channel size significantly and the last layer re-projects the lower dimensional representation back to the original dimensionality.

According to the paper, BERT model trained with the adapter method reaches a modeling performance comparable to a fully finetuned BERT model while only requiring the training of 3.6% of the parameters.
### Adapters vs.  Feature-Based Transfer and Fine-Tuning
Adapters are new modules added between layers of a pre-trained network. Adapter-based tuning differs from feature-based transfer and fine-tuning in the following way. Consider a function (neural network) with parameters $\boldsymbol w$: $\phi_{\boldsymbol w}(\boldsymbol x)$. 

- Feature-based transfer composes $\phi_{\boldsymbol w}$ with a new function, $\chi_{\boldsymbol v}$, to yield $\chi_{\boldsymbol v}(\phi_{\boldsymbol w}(\boldsymbol x))$. Only the new, task-specific, parameters, $\boldsymbol v$, are then trained. 
- Fine-tuning involves adjusting the original parameters, $\boldsymbol w$, for each new task, limiting compactness. 
- For adapter tuning, a new function, $\psi_{\boldsymbol w, \boldsymbol v}(\boldsymbol x)$, is defined, where parameters $\boldsymbol w$ are copied over from pre-training. The initial parameters $\boldsymbol v_0$ are set such that the new function resembles the original: $\psi_{\boldsymbol w, \boldsymbol v_0}(\boldsymbol x) \approx \phi_{\boldsymbol w}(\boldsymbol x)$. 

For adapter tuning training, only $\boldsymbol v$ are tuned. For deep networks, defining $\psi_{\boldsymbol w, \boldsymbol v}$ typically involves adding new layers to the original network, $\phi_{\boldsymbol w}$. If one chooses $|\boldsymbol v|\ll|\boldsymbol w|$, the resulting model requires $\sim|\boldsymbol w|$ parameters for many tasks. Since $\boldsymbol w$ is fixed, the model can be extended to new tasks without affecting previous ones.
