---
tags:
  - flashcards
source: https://arxiv.org/abs/2106.09685
summary: 
aliases:
  - Low-Rank Adaptation
  - Low-Rank Adaptation of Large Language Models
---
https://jaketae.github.io/study/lora/
[GitHub Post on Choosing the Rank](https://github.com/cloneofsimo/lora/discussions/37)
[Coffee Chat YouTube Video](https://youtu.be/KEv-F5UkhxU)

We often want to finetune base LLMs to use them on downstream tasks. However, it is difficult to finetune them since they contain billions of parameters and we also need to have a huge amount of memory to store both the model weights and the gradients for each weight. Using LoRA allows us to finetune with a lot fewer parameters.

LoRA freezes the original model weights during finetuning and instead adds a separate set of weights to train. After finetuning, these set of weights represent the differences we need to add to the pretrained parameters to make the model better for the finetuning task. At inference, you just load up the original weights + the updates and you still have the same number of parameters at inference.

LoRA is able to train fewer weights than the original weights of the model by ==decomposing the weight matrices into their low-rank decompositions== which use fewer parameters while still approximating the full matrices.
<!--SR:!2024-02-08,81,290-->

### Rank of a matrix
The rank of a matrix is the number of ==linearly independent== columns in the matrix.
![[lora-20231028094625618.png]]
A column is linearly independent if you can't create it from a combination of the other columns in the matrix.
<!--SR:!2024-03-14,109,290-->

### Low rank decomposition
LoRA's idea is that you don't need to optimize the full rank of the matrices. Instead, you can do a low-rank decomposition as representing the weights as a composition of two matrices ($Y$ and $Z$ in the diagram below):
![[screenshot 2023-10-28_09_50_18@2x.png]]
[Stanford Lecture Notes](https://web.stanford.edu/class/cs168/l/l9.pdf)

You can represent a matrix of size $m \times n$ with rank $k$ by two matrices of size $m \times k$ and $k \times n$. Instead of needing $m \cdot n$ parameters, you just need $k(m + n)$ parameters which is much smaller if the rank of the matrix is lower than the number of columns in it. For example, a $10 \times 12$ matrix with rank 4 will have $10 \cdot 12 = 120$ parameters but only $4(10 + 12) = 88$ parameters. Note: if the rank is not much lower than the number of columns, you don't see a saving of parameters:
```
# ROWS = 10, COLS = 12
Rank=1: Parameters=22
Rank=2: Parameters=44
Rank=3: Parameters=66
Rank=4: Parameters=88
Rank=5: Parameters=110
Rank=6: Parameters=132
Rank=7: Parameters=154
Rank=8: Parameters=176
Rank=9: Parameters=198
Rank=10: Parameters=220
Rank=11: Parameters=242
Rank=12: Parameters=264
```

> [!NOTE]
> The matrix $Y$ allows us to shrink the dimensionality of $A$ to just the $k$ linearly independent columns and then the matrix $Z$ allows us to recover the original number of columns of $A$.

### LoRA's approach
LoRA's learned weight differences $\Delta W$ ($d \times k$) are approximated by two matrices $A$ ($d \times r$) and $B$ ($r \times k)$ where $r$ is the rank of $\Delta W$.

$A$ is initialized from a Gaussian distribution and $B$ is initialized to $0$ and then you let backprop decide the parameter updates.

Instead of tuning the large matrix $\Delta W$, you finetune the smaller weight matrices $A$ and $B$.

After you find $\Delta W$, you add these deltas to the original weights to get the final weights.

> [!NOTE] How to choose the rank?
> The rank is a hyper-parameter and it seems even using rank = 1 can get good results. You can get away with very low ranks even if you end up losing a bit of information in your low rank decomposition. [GitHub Post](https://github.com/cloneofsimo/lora/discussions/37)
### LoRA vs. [[Parameter-Efficient Transfer Learning for NLP|Adapter Modules]]
Adapters add small learnable layers throughout the network that update the behavior of the model by finetuning. They are very compute efficient but large models are often parallelized on hardware. Adapter modules need to be processed sequentially.
### LoRA vs. [[Parameter Efficient Fine-Tuning|Prefix Tuning]]
Instead of manually choosing the right words to prompt the model, you use input vectors that are concatenated to the prompt and then tune these vectors using backprop until the model delivers the correct answer.

Instead of using words to prompt the model:
![[screenshot 2023-10-28_10_09_51@2x.png]]
you use input vectors that don't stand for any words in particular:
![[screenshot 2023-10-28_10_10_06@2x.png]]

The issue with prefix tuning is it occupies part of the sequence length and reduces the size of the effective input. It is also difficult to optimize and the number of trainable parameters is hard to choose.