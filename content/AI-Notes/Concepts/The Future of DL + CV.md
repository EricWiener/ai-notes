---
tags: [flashcards, eecs498-dl4cv]
source:
summary:
---
[[498_FA2019_lecture22.pdf]]

# Predictions
### We will discover new types of deep models.
- Neural ODE is a new type of model that represents neural networks as solutions to differential equations. You are able to represent a deep network with infinitely many layers.

### Deep Learning will find new applications
We will apply the ideas of deep learning applied to more scientific and medical applications.

**Hash Table:**
A traditional hash table has a hand-crafted function that assigns a particular hash for a key. You then insert the value depending on its hash and store it in a bucket. You want to reduce the number of hashes that go to the same bucket.

You can replace this hash function with a neural network:

![[AI-Notes/Concepts/the-future-of-dl+cv-srcs/Screen_Shot_1.png]]

### Deep Learning will use more data and more compute
We use larger amounts of computation for larger models trained on larger datasets.
![[AI-Notes/Concepts/the-future-of-dl+cv-srcs/Screen_Shot.png]]
Cerebras is a new type of chip that has a lot more transistors than a GPU

# Problems

### Bias

There is a bias in the training data, so there is a bias in the predictions.

### Need new theory

We need to understand what's going on inside the models. This is like before we knew quantum mechanics and we couldn't understand classical physics.

**Empirical Mystery:**

You can randomly initialize a network and train it. If you then remove weights of small magnitude, the pruned network works just about the same as the full network. It isn't clear why this is.

![[AI-Notes/Concepts/the-future-of-dl+cv-srcs/Screen_Shot 1.png]]

It gets even more confusing after this. You can take the pruned network and reset the remaining weights to be the values they were in the originally initialized network (the dropped weights remain dropped). Then, you can train this re-initialized, pruned network and you will get a model that works almost as well as the full network.

![[AI-Notes/Concepts/the-future-of-dl+cv-srcs/Screen_Shot 2.png]]

The **lottery ticket hypothesis** is that within a random deep network is a good subnetwork that won the "initialization lottery" and just happened to be initialized in a good way, which caused good subnetworks to develop.

You can even randomly initialize a network from scratch and then prune it to find an untrained subnetwork that works for classification. No training is needed and the subnetwork can already work as a classifier.

### Generalization
There is a bit of a mystery about why neural networks are able to generalize to testing data. If you take a model like ResNet50 and train it on data where the labels have been randomly shuffled (no longer correspond to the correct class), the model will be able to achieve zero training loss. 

Classical statical learning theory says if we can fit random data, then the model is likely to complex to generalize. However, ResNet50 is able to generalize and have great performance on test data.

### We need a lot of labelled training data
People have started making progress on this by building datasets that have very small number of examples per class.

![[AI-Notes/Concepts/the-future-of-dl+cv-srcs/Screen_Shot 3.png]]

![[AI-Notes/Concepts/the-future-of-dl+cv-srcs/Screen_Shot 4.png]]

![[AI-Notes/Concepts/the-future-of-dl+cv-srcs/Screen_Shot 5.png]]

Another approach is to use self-supervised learning. You pre-train the model on data that doesn't require labels. Then, you fine-tune the model on the target task without using that much labelled data. We have already seen something like this with generative networks, but it can also be seen with models that try to solve jigsaw puzzles. You shuffle up an image into patches and then have the model try to solve it. Ideally the model will be able to learn structure about the world.

![[AI-Notes/Concepts/the-future-of-dl+cv-srcs/Screen_Shot 6.png]]

### Deep Learning doesn't understand the world
Deep Learning models can mimic data that they are trained on, but they don't seem to understand the world. CNNs "see" the world in a different way from us. They can fail on images even slightly different from those seen during training.