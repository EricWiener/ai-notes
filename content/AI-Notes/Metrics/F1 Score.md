---
tags: [flashcards]
source:
aliases: [F1 Measure, Dice Loss, ]
summary: combines the precision and recall of a classifier into a single metric by taking their harmonic mean
---

$$F_1=\frac{2}{\frac{1}{\text { recall }} \times \frac{1}{\text { precision }}}=2 \times \frac{\text { precision } \times \text { recall }}{\text { precision }+ \text { recall }} =\frac{t p}{t p+\frac{1}{2}(f p+f n)} = \frac{2 T P}{2 T P+F P+F N}$$

> [!NOTE] F1 score is the same thing as F1 Measure and Dice Loss

### F1 Score for Segmentation
For segmentation problems it's easiest to consider the formulation $\frac{2 T P}{2 T P+F P+F N}$. If we were trying to segment a cat in an image that had a cat in the foreground and the remainder of the image were background, we would have the following:
- TP: correctly predicted cat pixels
- FP: background predicted as cat
- FN: cat predicted as background

For this example, the image consists of 500 pixels and 300 of those pixels are the cat. 
- If we perfectly predict the cat, we have $\frac{2 * 300}{2*300 + 0 + 0} = 1$.
- If we predict all background, we have $\frac{0}{0 + 0 + 300} = 0$.
- If we predict all cat, we have $\frac{2*300}{2*300 + 200 + 0} = \frac{600}{800}$.

When using cross entropy loss, the statistical distributions of labels play a big role in training accuracy. The more unbalanced the label distributions are, the more difficult the training will be. Although weighted cross entropy loss can alleviate the difficulty, the improvement is not significant nor the intrinsic issue of cross entropy loss is solved.

 In cross entropy loss, the loss is calculated as the average of per-pixel loss, and the per-pixel loss is calculated discretely, without knowing whether its adjacent pixels are boundaries or not. **As a result, cross entropy loss only considers loss in a micro sense rather than considering it globally, which is not enough for image level prediction.**

### Why does Dice Loss handle class imbalances better than cross entropy loss?
#### Explanation 1
Dice score measures the relative overlap between the prediction and the ground truth (intersection over union). It has the same value for small and large objects both: _Did you guess a half of the object correctly? Great, your loss is 1/2. I don't care if the object was 10 or 1000 pixels large._

On the other hand, cross-entropy is evaluated on individual pixels, so large objects contribute more to it than small ones, which is why it requires additional weighting to avoid ignoring minority classes.

A problem with dice is that it can have high variance. Getting a single pixel wrong in a tiny object can have the same effect as missing nearly a whole large object, thus the loss becomes highly dependent on the current batch. I don't know details about the generalized dice, but I assume it helps fighting this problem.

[Source](https://stats.stackexchange.com/q/438494/271266)

#### Explanation 2
We prefer Dice Loss instead of Cross Entropy because most of the semantic segmentation comes from an unbalanced dataset. Let me explain this with a basic example,  
Suppose you have an image of a cat and you want to segment your image as cat(foreground) vs not-cat(background).  
In most of these image cases you will likely see most of the pixel in an image that is not-cat. And on an average you may find that 70-90% of the pixel in the image corresponds to background and only 10-30% on the foreground.  
So, if one use CE loss the algorithm may predict most of the pixel as background even when they are not and still get low errors.  
But in case of Dice Loss ( function of Intersection and Union over foreground pixel ) if the model predicts all the pixel as background the intersection would be 0 this would give rise to error=1 ( maximum error as Dice loss is between 0 and 1).

What if I predict all the pixel as foreground ???  
In such case, the Union will become quite large and this will increase the loss to 1.

Hence, Dice loss gives low error as it focuses on maximising the intersection area over foreground while minimising the Union over foreground.

[Source](https://www.kaggle.com/getting-started/133156)

# F1 Score/Dice Score vs. IoU
https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou

# Using F1 Score for comparing different datasets 
![[AI-Notes/Metrics/f1-score-srcs/Untitled.png]]

F1-Measure is useful because it is hard to compare datasets when one has a high recall and a low specificity and the other has the opposite. The f1 score provides a balance between these two. It uses a harmonic mean.

![Arithmetic mean](https://i.stack.imgur.com/SYA7h.jpg)

Arithmetic mean

![Geometric mean](https://i.stack.imgur.com/slxFc.jpg)

Geometric mean

Source: [https://stackoverflow.com/a/49535509/6942666](https://stackoverflow.com/a/49535509/6942666)

*In order to have a high f1 score, both the recall and precision must be high. It will punish the f1 score more, the lower either is.*

## Harmonic Mean

Source: [https://towardsdatascience.com/on-average-youre-using-the-wrong-average-geometric-harmonic-means-in-data-analysis-2a703e21ea0](https://towardsdatascience.com/on-average-youre-using-the-wrong-average-geometric-harmonic-means-in-data-analysis-2a703e21ea0)

***Harmonic Means:** if the **reciprocal** of every number in our dataset **was the same number**, what number would it have to be **in order to have the same reciprocal sum** as our actual dataset?*

**Example:**

Consider a trip to the grocery store & back:

- On the way **there** you drove **30 mph** the entire way
- On the way **back** traffic was crawling, & you drove **10 mph** the entire way
- You took the same route, & covered the same amount of ground (**5 miles**) each way.

What was your average speed across this entire trip’s duration?

Again, we might naively apply the **arithmetic mean** to **30 mph** & **10 mph**, and proudly declare “**20 mph**!”

**Trip There: (at 30 mph)**

`30 miles per 60 mins = 1 mile every 2 minutes = 1/2 mile every minute`

`5 miles at 1/2 mile per minute = 5 ÷ 1/2 = 10 minutes`

**`"Trip There" time** = **10 minutes**`

**Trip Back: (at 10 mph)**

`10 miles per 60 mins = 1 mile every 6 minutes = 1/6 miles every minute`

`5 miles at 1/6 mile per minute = 5 ÷ 1/6 = 30 minutes`

**`"Trip Back" time** = **30 minutes**`

**`Total trip time** = 10 + 30 = **40 minutes**`

**`“Trip There” % of total trip** = 10 / 40 minutes = .25 = 25%`

**`“Trip Back” % of total trip** = 30 / 40 minutes = .75 = 75%`

**`Weighted Arithmetic Mean** = (30mph * .25)+(10mph * .75) = 7.5 + 7.5 = 15`

**`Average rate of travel = 15 mph`**

So we see that our true average rate of travel was **15 mph**, which is 5 mph (or 25%) lower than our naive declaration of **20 mph** using an unweighted **arithmetic mean**.

You can probably guess where this is headed…

Let’s try it again using the **harmonic mean**.

**`Harmonic mean** of 30 and 10 = ...`

**`Arithmetic mean** of **reciprocals** = 1/30 + 1/10 = 4/30 ÷ 2 = 4/60 = 1/15`

**`Reciprocal** of **arithmetic mean** = 1 ÷ 1/15 = 15/1 = **15**`

*The **harmonic mean** is equivalent to the **weighted arithmetic mean** of rate of travel (where values are weighted by relative time spent traveling)*

*To average rates over different periods or lengths: use the **harmonic mean** (or **weighted arithmetic mean**)*