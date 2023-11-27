---
tags: [flashcards]
source: [eecs498-dl4cv]
summary:
---
## Supervised vs. Unsupervised
**Supervised Learning:**
- With supervised learning, we have data and labels for the data
- We want to learn a function to map $x \rightarrow y$.
- Examples: classification, regression, object detection, semantic segmentation, image captioning, etc.
- Need humans to annotate the outputs that you want for your inputs. You need a lot of labelled data.

**Unsupervised Learning:**
- This is the catch-all term for everything that isn't supervised
- You get the data $x$, but you don't get the labels $y$.
- You want to build a model that can uncover hidden structure in the data.
- Examples: clustering, dimensionality reductions ([[Principal Component Analysis|PCA]]), feature learning (e.g. [[Autoencoders]] where you want to learn some underlying hidden structure of the data), density estimation, etc.
- Want to be able to just take in data and have the model become better and better.

# Discriminative vs. Generative Models
### Density Function Refresher
A density function $p(x)$ assigns a positive number to each possible $x$. A higher number means that $x$ is more likely. Density functions are normalized so that they add up to 1:

$$
\int_x p(x)dx = 1
$$

This means that different values of $x$ compete for density, since everything needs to add up to 1.

**Note:** when dealing with infinite distributions (ex. all images), you can't talk about the probability of a certain point, since it is infinitely small. Instead, you need to integrate over a range centered at that point in order to talk about probability.

## Model Comparison
### **Discriminative Model:**
Learn a probability distribution $p(y|x)$. Tries to predict a label conditioned on the input image. This is what an image classifier does. This is typically supervised learning.

![[Supervised/Screen_Shot.png]]

![[Supervised/Screen_Shot 1.png]]

With a discriminative models, the **different labels for each input compete** for probability mass, but there is **no competition between images.**

![[Unreasonable input for a discriminative model]]

Unreasonable input for a discriminative model

If you introduce an input that doesn't match the labels, the model will be forced to assign it probabilities from the known labels. There is no way for the model to handle unreasonable inputs.

**What you can do:**

- You can assign labels to data
- Feature learning (with labels)

### **Generative Model:**

Learn a probability distribution $p(x)$ which tells us something about the data. For instance, this would try to learn something about images $x$. This is typically unsupervised learning.

![[Supervised/Screen_Shot 3.png]]

With a generative model, all the **possible images compete** with each other for probability mass. The model needs to have a very good understanding of the images in order to decide how likely something is.

The model can "reject" unreasonable inputs by assigning them small values.

In order to evaluate how well a generative model performs, you can give it unseen images and see if it gives a high probability density if they are similar to the images it has already seen.

**What you can do:**

- Detect outliers (detect data different from data it was trained on)
- Feature learning (without labels)
- Sample from the model to generate new data

### **Conditional Generative Model:**

Want to learn $p(x|y)$. For instance this would try to learn something about images $x$ that are of type $y$. This typically is also supervised learning because you need both the data and the labels.

For every possible label $y$, it will learn a distribution over the data $x$. Each possible label induces a competition among all images.

![[The top distribution is conditioned on the image being of a cat. The second row is conditioned on the image being a dog.]]

The top distribution is conditioned on the image being of a cat. The second row is conditioned on the image being a dog.

You could feasibly train an image classifier using a conditional generative model. The image would have a different likelihood in every label's distribution. You would just see for which label it had the highest distribution.

The conditional generative model is able to reject an image unlike the discriminative model. This is because an image can receive a low likelihood in the distribution for every label. You can set a threshold where you refuse to classify something. The discriminative model, on the other hand, needed to give one class a higher score. 

You can build a conditional generative model from the discriminative and (unconditional) generative model using Bayes' Rule. This means we can focus on building these two models and we can build other models from these.

![[Supervised/Screen_Shot 5.png]]

**What you can do:**

- Assign labels
- Detect outliers (detect data different from data it was trained on)
- Generate new data conditioned on input labels (generate new images of dogs).