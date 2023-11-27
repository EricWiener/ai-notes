---
aliases:
source: EECS 498 DL4CV Lecture 19 - Generative Models
tags: [eecs498-dl4cv, flashcards]
summary:
---

# Discriminative vs. Generative Models
### Density Function
A density function $p(x)$ assigns a positive number to each possible $x$. A higher number means that $x$ is more likely. Density functions are normalized so that they add up to 1 when you integrate over all possible $x$:

$$
\int_x p(x)dx = 1
$$

This means that different values of $x$ compete for density, since everything needs to add up to 1.

**Note:** when dealing with infinite distributions (ex. all images), you can't talk about the probability of a certain point, since it is infinitely small. Instead, you need to integrate over a range centered at that point in order to talk about probability. If you want to talk about a specific point, you need to use **likelihood.** 

## Model Comparison
There are three main types of models:
- Discriminative model: learn $p(y|x)$. Predict label $y$ conditioned on input image $x$. Summing over label probabilities = 1.
- Generative model: learn $p(x)$. Predict probability of input image $x$ given distribution of training data. Summing over all possible inputs = 1.
- Conditional generative model: $p(x|y)$. Predict probability of an input image $x$ given label $y$. Summing over all possible inputs = 1 for each label $y$.

### **Discriminative Model:**
Learn a probability distribution $p(y|x)$. Tries to predict the probability of a label $y$ conditioned on the input image $x$. This is what an image classifier does. This is typically supervised learning.

![[AI-Notes/Generative/supervised-vs.-unsupervised-learning-srcs/Screen_Shot.png|300]] ![[AI-Notes/Generative/supervised-vs.-unsupervised-learning-srcs/Screen_Shot 1.png|300]]

With a discriminative models, the **different labels for each input compete** for probability mass, but there is **no competition between images.**

**Downside**: you can't handle unreasonable inputs. If you introduce an input that doesn't match the labels, the model will be forced to assign it probabilities from the known labels.
![[unreasonable-input-for-discriminative-model.png]]
Unreasonable input for a discriminative model

**What you can do with discriminative models:**
- You can assign labels to data
- Feature learning (with labels). Ex: pre-training on ImageNet to learn features and then fine-tuning on a different dataset.

### **Generative Model:**
Learn a probability distribution $p(x)$ which tells us something about the data. For instance, this would try to learn something about images $x$ and then you could say how likely you would see a particular image given the training distribution. This is typically unsupervised learning.
![[AI-Notes/Generative/supervised-vs.-unsupervised-learning-srcs/Screen_Shot 3.png]]

With a generative model, all the **possible images compete** with each other for probability mass. The model needs to have a very good understanding of the images in order to decide how likely something is.

The model can "reject" unreasonable inputs by assigning them small values.

In order to evaluate how well a generative model performs, you can give it unseen images and see if it gives a high probability density if they are similar to the images it has already seen.

**What you can do with generative models:**
- Detect outliers (detect data different from data it was trained on)
- Feature learning (without labels). Ex: you have access to a large set of images without labels. You could try to to learn a generative model and hopefully this model will be useful when transferred to downstream tasks.
- Sample from the model to generate new data.

### **Conditional Generative Model:**
Want to learn $p(x|y)$. For instance this would try to learn something about images $x$ that are of type $y$. This typically is also supervised learning because you need both the data and the labels.

For every possible label $y$, it will learn a distribution over the data $x$. Each possible label induces a competition among all images.

![[AI-Notes/Generative/supervised-vs.-unsupervised-learning-srcs/Screen_Shot 4.png]]
The top distribution is conditioned on the image being of a cat. The second row is conditioned on the image being a dog. The distributions for conditioned on both cat and dog sum to 1.

You could feasibly train an image classifier using a conditional generative model. The image would have a different likelihood in every label's distribution. You would just see for which label it had the highest distribution.

The conditional generative model is able to reject an image unlike the discriminative model. This is because an image can receive a low likelihood in the distribution for every label. You can set a threshold where you refuse to classify something. The discriminative model, on the other hand, needed to give one class a higher score. 

**Building conditional generative model from discrimative and (unconditional) generative model:**
You can build a conditional generative model from the discriminative and (unconditional) generative model using Bayes' Rule. In principal this means you could build these two models and you can build other models from these, but this is not often done in practice (you usually build conditional generative model directly).
![[build-conditional-generative-from-discriminative-and-generative.png]]
$P(y)$ is known because you just look at your training data and for each label $y$ you look how often it ocurred / total number of examples.

**What you can do with conditional generative models:**
- Assign labels (use the label that has the highest conditional probability associated with it).
- Detect outliers (detect data different from data it was trained on) by assigning a low conditional probability for all labels.
- Generate new data conditioned on input labels (generate new images of a particular label vs. a regular generative model which just generates something randomly from the learned distribution).