---
tags: [flashcards]
source: https://twitter.com/karpathy/status/1013244313327681536?lang=en
summary: most common ML mistakes according to Andrej Karpathy
---

1) You didn't try to overfit a single batch first.
2) You forgot to toggle train/eval mode for the net.
3) You forgot to .zero_grad() (in pytorch) before .backward().
4) You passed softmaxed outputs to a loss that expects raw logits (ex. `BCEWithLogitsLoss` expects logits while `BCELoss` expects outputs of `softmax`).
5) You didn't use bias=False for your Linear/Conv2d layer when using BatchNorm, or conversely forget to include it for the output layer. This one won't make you silently fail, but they are spurious parameters. See [[Batch Normalization#You don't need a bias term for preceeding layer when using BatchNorm]].
6) Thinking view() and permute() are the same thing (& incorrectly using view). 

### The difference between view and permute
```python
x = torch.arange(2*4).view(2, 4)  
print(x.view(4, 2))  
> tensor([[0, 1],  
          [2, 3],  
          [4, 5],  
          [6, 7]])  
print(x.permute(1, 0))  
> tensor([[0, 4],  
          [1, 5],  
          [2, 6],  
          [3, 7]])
```