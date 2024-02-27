---
tags:
  - flashcards
source: https://pair.withgoogle.com/explorables/private-and-fair/
summary: "Training with differential privacy limits the information about any one data point that is extractable but in some cases there’s an unexpected side-effect: reduced accuracy with underrepresented subgroups disparately impacted."
publish: true
---

> [!NOTE]
> The following contains excerpts and diagrams from three great articles:
> - [Pair with Google](https://pair.withgoogle.com/explorables/private-and-fair/)
> - [Differential Privacy Medium Article](https://medium.com/@shaistha24/differential-privacy-definition-bbd638106242)
> - [DifferentialPrivacy.org](https://differentialprivacy.org/flavoursofdelta/)

Imagine you want to use machine learning to suggest new bands to listen to. You could do this by having lots of people list their favorite bands and using them to train a model. The trained model might be quite useful and fun, but if someone pokes and prods at the model in just the right way, they could [extract](https://www.wired.com/2007/12/why-anonymous-data-sometimes-isnt/) the music preferences of someone whose data was used to train the model. Other kinds of models are potentially vulnerable; [credit card numbers](https://bair.berkeley.edu/blog/2019/08/13/memorization/) have been pulled out of language models and [actual faces](https://rist.tech.cornell.edu/papers/mi-ccs.pdf) reconstructed from image models.

Training with [differential privacy](https://desfontain.es/privacy/differential-privacy-awesomeness.html) limits the information about any one data point that is extractable but in some cases there’s an unexpected side-effect: reduced accuracy with underrepresented subgroups disparately impacted.

Recall that machine learning models are typically trained with [gradient descent](https://playground.tensorflow.org/), a series of small steps taken to minimize an error function. To show how a model can leak its training data, we’ve trained two simple models to separate red and blue dots using two simple datasets that differ in one way: a single isolated data point in the upper left has been switched from red to blue.
![[AI-Notes/assets/Differentially Private Training/screenshot_2024-02-20_11_25_32@2x.png|200]]
We can prevent a single data point from drastically altering the model by [adding](http://www.cleverhans.io/privacy/2019/03/26/machine-learning-with-differential-privacy-in-tensorflow.html) two operations to each training step:[¹](https://pair.withgoogle.com/explorables/private-and-fair/#footend-0)
- Clipping the gradient (here, limiting how much the boundary line can move with each step) to bound the maximum impact a single data point can have on the final model.
- Adding random noise to the gradient.
![[AI-Notes/assets/Differentially Private Training/screenshot_2024-02-20_11_33_23.gif|400]]

In [Distribution Density, Tails, and Outliers in Machine Learning](https://arxiv.org/abs/1910.13427), a series of increasingly differentially private models were trained on [MNIST digits](https://en.wikipedia.org/wiki/MNIST_database). Every digit in the training set was ranked according to the highest level of privacy that correctly classified it.

![[AI-Notes/assets/Differentially Private Training/screenshot_2024-02-20_11_35_12@2x.png|400]]

On the lower left, you can see digits labeled as “3” in the training data that look more like a “2” and a “9”. They’re very different from the other “3”s in the training data so adding just a bit of privacy protection causes the model to no longer classify them as “3”. Under some [specific circumstances](https://arxiv.org/abs/1411.2664), differential privacy can actually improve how well the model generalizes to data it wasn’t trained on by limiting the influence of spurious examples.

### Increasing privacy can result in poor representation for minority classes
It’s possible to inadvertently train a model with good overall accuracy on a class but very low accuracy on a smaller group within the class.

This disparate impact doesn’t just happen in datasets of differently drawn digits: increased levels of differential privacy in a range of image and language models disproportionality decreased accuracy on underrepresented subgroups. And adding differential privacy to a medical model reduced the influence of Black patients’ data on the model while increasing the influence of white patients’ data.

Lowering the privacy level might not help non-majoritarian data points either – they’re the ones most susceptible to having their information exposed. Again, escaping the accuracy/privacy tradeoff requires collecting more data – this time from underrepresented subgroups.

### ε-differential privacy
[This](https://medium.com/@shaistha24/differential-privacy-definition-bbd638106242) is a great article.

**Formal Definition of Differential Privacy:**
![[AI-Notes/assets/Differentially Private Training/image-20240220122433678.png|500]]
Where:
**M:** Randomized algorithm i.e `query(db) + noise` or `query(db + noise).`
**S:** All potential output of M that could be predicted.
**x:** Entries in the database. (i.e, N)
**y:** Entries in parallel database (i.e., N-1)
**ε:** The maximum distance between a query on database (x) and the same query on database (y).
**δ:** Probability of information accidentally being leaked.

> [!TLDR] $\epsilon$ differential privacy
> - It gives the comparison between running a query M on database (x) and a parallel database (y) (One entry less than the original database).
> - The measure by which these two probability of random distribution of full database(x) and the parallel database(y) can differ is given by Epsilon (ε) and delta (δ).
> - **_Small values_** _of ε require to provide_ **very _similar outputs_** _when given_ **_similar inputs_**_, and therefore_ **_provide higher levels of privacy_**_;_ **_large values_** _of ε allow_ **_less similarity_** _in the outputs, and therefore provide_ **_less privacy_**_._

### $(\epsilon, \delta)$ differential privacy
[This](https://differentialprivacy.org/flavoursofdelta/) is a great article. “Pure” DP has a single parameter $\epsilon$, and corresponds to a very stringent notion of DP:
> An algorithm $M$ is $\varepsilon-D P$ if, for all neighboring inputs $D, D^{\prime}$ and all measurable $S, \operatorname{Pr}[M(D) \in S] \leq e^{\varepsilon} \operatorname{Pr}\left[M\left(D^{\prime}\right) \in S\right]$.

By relaxing this a little, one obtains the standard definition of approximate DP, a.k.a. $(\epsilon, \delta)$-DP:
> An algorithm $M$ is $(\varepsilon, \delta)-D P$ if, for all neighbouring inputs $D, D^{\prime}$ and all measurable $S, \operatorname{Pr}[M(D) \in S] \leq e^{\varepsilon} \operatorname{Pr}\left[M\left(D^{\prime}\right) \in S\right]+\delta$.

This definition is very useful, as in many settings achieving the stronger $\epsilon$-DP guarantee (i.e., $\delta$=0) is impossible, or comes at a very high utility cost. But how to interpret it? Most of the time, things are great, but with some small probability $\delta$ all hell breaks loose. For instance, the following algorithm is $(\epsilon, \delta)$-DP:
- Get a sensitive database $D$ of $n$ records
- Select uniformly at random a fraction $\delta$ of the database ($\delta n$ records)
- Expose that subset of records publicly and delete all other records (all hell breaks loose for those exposed records💥)

This approach is $(\epsilon = 0, \delta)$ since most of the time there is 0 probability of a record being exposed since it was deleted. However, there is a $\delta$ probability of a record being exposed.