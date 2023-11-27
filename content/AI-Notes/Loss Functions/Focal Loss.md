---
tags: [flashcards]
source: https://web.archive.org/web/20220413131232/https://amaarora.github.io/2020/06/29/FocalLoss.html
summary: what is Focal Loss and when is it used.
---

#### TLDR
Focal Loss **modifies Cross Entropy Loss to improve performance on hard to predict objects and imbalanced classes**.
$$\mathrm{FL}\left(p_{\mathrm{t}}\right)=-\alpha_{\mathrm{t}}\left(1-p_{\mathrm{t}}\right)^\gamma \log \left(p_{\mathrm{t}}\right)$$
$(1-p_{\mathrm{t}})$ will decrease the loss for predictions with higher probability so the loss focuses more on predictions with lower confidence (focus on 10% -> 40% vs. trying to push 80% confidence to 90%).

$\alpha_{\mathrm{t}}$ is a per-class weight to help deal with class imbalances. Note that implementations for binary focal loss use a scalar while multi-class implementations use a tensor of length NUM_CLASSES.

#### Understanding the equation
**What is the point of the $\alpha_t$ term in Focal Loss $\mathrm{FL}\left(p_{\mathrm{t}}\right)=-\alpha_{\mathrm{t}}\left(1-p_{\mathrm{t}}\right)^\gamma \log \left(p_{\mathrm{t}}\right)$?**
??
It is a per-class weight to help deal with class imbalances.
<!--SR:!2025-07-25,776,330-->

**What is the point of $(1-p_{\mathrm{t}})$ term in Focal Loss $\mathrm{FL}\left(p_{\mathrm{t}}\right)=-\alpha_{\mathrm{t}}\left(1-p_{\mathrm{t}}\right)^\gamma \log \left(p_{\mathrm{t}}\right)$?**
??
$(1-p_{\mathrm{t}})$ will decrease the loss for predictions with higher probability so the loss focuses more on predictions with lower confidence (focus on 10% -> 40% vs. trying to push 80% confidence to 90%).
<!--SR:!2024-02-19,135,290-->

**What is the affect of different $\gamma$ values?**
The image below shows how Binary Focal Loss behaves for different values of $\gamma$. The higher the gamma, the lower the loss will be for correctly classified predictions with higher confidence. When $\gamma = 0$, this is the same as `BCELoss`.
![[focal-loss-w-different-gamma-values.png]]

## Where was Focal Loss introduced and what was it used for?

Before understanding what Focal Loss is and all the details about it, let’s first quickly get an intuitive understanding of what Focal Loss actually does. Focal loss was implemented in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) paper by He et al.

For years before this paper, Object Detection was actually considered a very difficult problem to solve and it was especially considered very hard to detect small size objects inside images. 

This is because this model was trained using [[Cross Entropy Loss]] which really asks the model to be confident about what is predicting. Whereasm, what Focal Loss does is that it makes it easier for the model to predict things without being 80-100% sure that this object is “something”. In simple words, giving the model a bit more freedom to take some risk when making predictions. This is particularly important when dealing with highly imbalanced datasets because in some cases (such as cancer detection), we really need to model to take a risk and predict something even if the prediction turns out to be a False Positive.

Therefore, Focal Loss is particularly useful in cases where there is a class imbalance. Another example, is in the case of Object Detection when most pixels are usually background and only very few pixels inside an image sometimes have the object of interest.

OK - so focal loss was introduced in 2017, and is pretty helpful in dealing 

## So, why did that work? What did Focal Loss do to make it work?
So now that we have seen an example of what Focal Loss can do, let’s try and understand why that worked. The most important bit to understand about Focal Loss is the graph below:

![[focal-loss-vs-cross-entropy.png]]
Fig 3: Comparing Focal Loss with Cross Entropy Loss

In the graph above, the “blue” line represents the **Cross Entropy Loss**. The X-axis or ‘probability of ground truth class’ (let’s call it `pt` for simplicity) is the probability that the model predicts for the ground truth object. As an example, let’s say the model predicts that something is a bike with probability 0.6 and it actually is a bike. The in this case `pt` is 0.6. Also, consider the same example but this time the object is not a bike. Then `pt` is 0.4 because ground truth here is 0 and probability that the object is not a bike is 0.4 (1-0.6).

The Y-axis is simply the loss value given `pt`.

As can be seen from the image, when the model predicts the ground truth with a probability of 0.6, the C**ross Entropy Loss** is still somewhere around 0.5. Therefore, to reduce the loss, our model would have to predict the ground truth label with a much higher probability. In other words, **Cross Entropy Loss** asks the model to be very confident about the ground truth prediction.

This in turn can actually impact the performance negatively:

> [!NOTE] The Deep Learning model can actually become overconfident and therefore, the model wouldn’t generalize well.

This problem of overconfidence is also highlighted in this excellent paper [Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration](https://arxiv.org/abs/1910.12656). Also, Label Smoothing which was introduced as part of [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) is another way to deal with the problem.

Focal Loss is different from the above mentioned solutions. As can be seen from the graph `Compare FL with CE`, using Focal Loss with γ>1 reduces the loss for “well-classified examples” or examples when the model predicts the right thing with probability > 0.5 whereas, it increases loss for “hard-to-classify examples” when the model predicts with probability < 0.5. Therefore, it turns the models attention towards the rare class in case of class imbalance.

The Focal Loss is mathematically defined as:
$$\mathrm{FL}\left(p_{\mathrm{t}}\right)=-\alpha_{\mathrm{t}}\left(1-p_{\mathrm{t}}\right)^\gamma \log \left(p_{\mathrm{t}}\right)$$

## Alpha and Gamma?

So, what the hell are these `alpha` and `gamma` in Focal Loss? Also, we will now represent `alpha` as `α` and `gamma` as `γ`.

Here is my understanding from fig-3:

> [!NOTE] `γ` controls the shape of the curve. The higher the value of `γ`, the lower the loss for well-classified examples, so we could turn the attention of the model more towards ‘hard-to-classify examples. Having higher `γ` extends the range in which an example receives low loss.

Also, when `γ=0`, this equation is equivalent to Cross Entropy Loss. How? Well, for the mathematically inclined, Cross Entropy Loss is defined as equation 2:
$$\mathrm{CE}(p, y)= \begin{cases}-\log (p) & \text { if } y=1 \\ -\log (1-p) & \text { otherwise }\end{cases}$$

After some refactoring and defining `pt` (probability of ground truth) as equation 3:
$$p_{\mathrm{t}}= \begin{cases}p & \text { if } y=1 \\ 1-p & \text { otherwise }\end{cases}$$

Putting `eq-3` in `eq-2`, our Cross Entropy Loss therefore, becomes equation 4:
$$\mathrm{CE}\left(p_{\mathrm{t}}\right)=-\log \left(p_{\mathrm{t}}\right)$$

Therefore, at `γ=0`, `eq-1` becomes equivalent to `eq-4` that is Focal Loss becomes equivalent to Cross Entropy Loss. [Here](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a) is an excellent blogpost that explains Cross Entropy Loss.

Ok, great! So now we know what `γ` does, but, what does `α` do?

Another way, apart from Focal Loss, to deal with class imbalance is to introduce weights. Give high weights to the rare class and small weights to the dominating or common class. These weights are referred to as `α`. Equation 5 (Alpha Weighted Cross Entropy Loss):
$$\mathrm{CE}\left(p_{\mathrm{t}}\right)=-\alpha_{\mathrm{t}} \log \left(p_{\mathrm{t}}\right)$$

Adding these weights does help with class imbalance however, the focal loss paper reports:

> [!NOTE] The large class imbalance encountered during training of dense detectors overwhelms the cross entropy loss. Easily classified negatives comprise the majority of the loss and dominate the gradient. While α balances the importance of positive/negative examples, it does not differentiate between easy/hard examples.

What the authors are trying to explain is this:

> [!NOTE] Even when we add α, while it does add different weights to different classes, thereby balancing the importance of positive/negative examples - just doing this in most cases is not enough. What we also want to do is to reduce the loss of easily-classified examples because otherwise these easily-classified examples would dominate our training.

So, how does Focal Loss deal with this? It adds a multiplicative factor to Cross Entropy loss and this multiplicative factor is `(1 − pt)**γ` where `pt` as you remember is the probability of the ground truth label.

From the paper for Focal Loss:

> [!NOTE] We propose to add a modulating factor (1 − pt)\*\*γ to the cross entropy loss, with tunable focusing parameter γ ≥ 0.

Really? Is that all that the authors have done? That is to add `(1 − pt)**γ` to Cross Entropy Loss? Yes!! Here is the Alpha Weighted Focal Loss:
$$\mathrm{FL}\left(p_{\mathrm{t}}\right)=-\alpha_{\mathrm{t}}\left(1-p_{\mathrm{t}}\right)^\gamma \log \left(p_{\mathrm{t}}\right)$$