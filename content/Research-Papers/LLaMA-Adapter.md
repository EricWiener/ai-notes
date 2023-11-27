---
tags:
  - flashcards
source: https://arxiv.org/abs/2303.16199
summary: Lightweight adaption method that efficiently finetunes LLaMA into an instruction-following model.
aliases:
---
This is a form of [[Parameter Efficient Fine-Tuning|PEFT]] that finetunes LLaMA to be an instruction following model using:
- A version of [[Parameter Efficient Fine-Tuning|Prefix Tuning]] that has separate weights for each transformer block
- Only modifying the last few transformer blocks
- Using a zero-initialized gating mechanism to stabilize training.
- And training on the 52k [[Self-Instruct]] dataset.

The main benefits of the paper are:
- They can outperform [[Alpaca]] (which finetuned LLaMA with all the weights unfrozen) with a much cheaper finetuning (one hour on 8 A100 GPUs) because they only learn 1.2M parameters for the adaption prompts instead of the full 7B parameters.
- You can swap out adapter weights for different tasks.
- You can use the adapter to also take images as input for multi-modal reasoning.
- Zero-initialized attention can be generalized to other models like traditional vision tasks (ex. finetuning a ViT for image classification).

# TLDR from [Lightning AI](https://lightning.ai/pages/community/article/understanding-llama-adapters/)
This paper builds on the ideas of [[Parameter Efficient Fine-Tuning|Prefix Tuning]] and [[Parameter-Efficient Transfer Learning for NLP|Adapter Modules]].

Like _prefix tuning_, the LLaMA-Adapter method prepends tunable prompt tensors to the embedded inputs. It’s worth noting that in the LLaMA-Adapter method, the prefix is learned and maintained within an embedding table rather than being provided externally. Each transformer block in the model has its own distinct learned prefix, allowing for more tailored adaptation across different model layers.

In addition, LLaMA-Adapter introduces a zero-initialized attention mechanism coupled with gating. The motivation behind this so-called _zero-init_ attention and gating is that adapters and prefix tuning could potentially disrupt the linguistic knowledge of the pretrained LLM by incorporating randomly initialized tensors (prefix prompts or adapter layers), resulting in unstable finetuning and high loss values during initial training phases.

Another difference compared to prefix tuning and the original adapter method is that LLaMA-Adapter adds the learnable adaption prompts only to the _L_ topmost (last L in the forward pass) transformer layers instead of all transformer layers. The authors argue that this approach enables more effective tuning of language representations focusing on higher-level semantic information.

![[llama-adapter-20231010093812430.png]]

# Related Work
Most existing approaches finetune pre-trained LLMs by using high-quality instruction-output data pairs. 

[[Alpaca]] fine-tunes llama, but unfreezes all 7B parameters. This is inefficient in both time and memory.

# Approach
In LLaMA's last transformer layers, they append a set of learnable adaption prompts as a prefix to the input instruction tokens + already generated token sequence. These prompts learn to inject new instructions into the frozen LLaMA model and you can use a different set of prompts for each task.

To avoid noise from adaption prompts at the early training stage, they modify the vanilla attention mechanisms at inserted layers to be zero-initialized attention, with a learnable gating factor. Initialized by zero vectors, the gating can firstly preserve the original knowledge in LLaMA, and progressively incorporate instructional signals during training.
### Learnable Adaption Prompts
They use a $N$ layer transformer and add adaption prompts to the last $L$ layers. They denote the learnable [[Parameter Efficient Fine-Tuning|soft prompts]] as $\{ P_l \}^L_{l = 1}$ where each prompt is a learnable tensor of shape $K \times C$ where $K$ is the prompt length (the same for each layer) and $C$ is the feature dimension of LLaMA's transformer.

Using the $l$-th inserted layer as an example, it will operate on a sequence of $M$ word tokens (the input instruction and the already generated response). They denote the sequence of tokens as $T_l \in \mathbb{R}^{M \times C}$ (it has the same channel dimension as the transformer).

> [!NOTE] What is $M$?
> $M$ is the sequence of word tokens the transformer is currently working on.
> 
> $M$ varies because the transformer predicts one token at a time so when you start predicting a response, $M$ will just be the length of the instruction and then each time a new token is predicted, $M$ will increase by 1.

The learnable adaption prompt for the $l$-th layer (denoted $P_l \in \mathbb{R}^{K \times C}$) is concatenated with $T_l$ (the input instruction + already generated response) and is formulated as:
$$\left[P_l ; T_l\right] \in \mathbb{R}^{(K+M) \times C}$$

This allows the instruction knowledge learned by the soft prompt prefix to guide $T_l$ (the input + currently generated response) to generate the subsequent contextual response via attention layers in the transformer block.

### Zero initialized attention
If the adaption prompts are randomly initialized, this might harm the fine-tuning stability at the beginning of training.

To address this, they modify the attention in the last $L$ transformer layers to be zero-initialized attention with a learnable gating factor $g_l$ to adaptively control how much the prompt is used. They use a separate $g_l$ for each head within each transformer's MHSA block.
![[llama-adapter-zero-init-attention.excalidraw|700]]

Suppose model is generating the $M + 1$-th word on top of $[P_l;T_l]$ at the $l$-th inserted layer. The corresponding $(M+1)$-th token is denoted $t_l \in \mathbb{R}^{1 \times C}$.

In the attention mechanism, you first convert get queries, keys, and values via linear projections:
- $Q_l=\operatorname{Linear}_{\mathrm{q}}\left(t_l\right)$: the query is just the single token being generated since at inference the model is [[Autoregressive]]. The token comes from the output of the previous layer's prediction
- $K_l=\operatorname{Linear}_{\mathrm{k}}\left(\left[P_l ; T_l ; t_l\right]\right)$: keys are calculated using the learnable prompt, already generated tokens, and token being generated
- $V_l=\operatorname{Linear}_{\mathrm{v}}\left(\left[P_l ; T_l ; t_l\right]\right)$: values are calculated using the learnable prompt, already generated tokens, and token being generated

You then calculate the attention scores of $Q_l$ and $K_l$ before softmax as:
$$S_l=Q_l K_l^T / \sqrt{C} \in \mathbb{R}^{1 \times(K+M+1)}$$
which is the similarity between the new word $t_l$ and all $K + M + 1$ tokens.

$S_l$ can be decomposed into two components:
$$S_l=\left[S_l^K ; S_l^{M+1}\right]^T$$
where $S_l^K \in \mathbb{R}^{K \times 1}$ and $S_l^{M+1} \in \mathbb{R}^{(M+1) \times 1}$ are the attention scores of the $K$ adaption prompts and $M + 1$ word tokens respectively. $S_l^K$ is how much information the learnable prompt contributes to generating $t_l$ (the output token prediction for this layer) which probably causes disturbance in the earlier layers.

To avoid the possible disturbance, they add a gating factor $g_l$ which is initialized as zero. They then independently apply the softmax function to the two components of $S_l$ and multiply the term corresponding to the learnable prompt's contribution by $g_l$:
$$S_l^g=\left[\operatorname{softmax}\left(S_l^K\right) \cdot g_l ; \operatorname{softmax}\left(S_l^{M+1}\right)\right]^T$$

$\operatorname{softmax}(S_l^{M+1})$ is the original attention behavior and since the softmax doesn't include $S_l^K$, the prompt has no affect on this term. $\operatorname{softmax}\left(S_l^K\right) \cdot g_l$ starts out as zero so the initial affect of this term will be zero and most of the pre-trained knowledge of LLaMA will be used.

Finally, the output of the $l$-th attention layer is calculated via a linear projection after multiplying and summing the attention weights and values:
$$t_l^o=\operatorname{Linear}_{\mathrm{o}}\left(S_l^g V_l\right) \in \mathbb{R}^{1 \times C}$$
# Multi-modal reasoning
LLaMA-Adapter can answer a question based on input of other modalities which augments the language model with cross-modal information.

They use a pre-trained visual encoder (ex. [[CLIP Learning Transferable Visual Models From Natural Language Supervision|CLIP]]) to extract multi-scale global features, concatenate the features along the channel dimension, and then project the concatenated features to $I_p \in \mathbb{R}^{1 \times C}$ which is the overall image token with the same feature dimension as the adaption prompts and the same channels as the transformer.

They then repeat $I_p$ $K$ times (the length of the learnable prompt) and element-wisely add it to the adaption prompts at all $L$ inserted transformer layers. They denote this:
$$P_l^v=P_l+\operatorname{Repeat}\left(I_p\right) \in \mathbb{R}^{K \times C}$$
where $P_l^v$ is the adaption prompt incorporating visual information from the given image.

![[llama-adapter-20231012112137832.png]]
