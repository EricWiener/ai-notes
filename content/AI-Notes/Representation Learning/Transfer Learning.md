---
tags: [flashcards, eecs442, eecs498-dl4cv]
source:
summary: Use a pre-trained model to solve a completely different task.
---
[[EECS 498 DL4CV/eecs-498-training-neural-networks-dynamics-srcs/498_FA2019_lecture11.pdf]]

We can use a pre-trained model to solve a completely different task. You don't actually need large datasets if you have pre-trained models.

Transfer learning can help you converge faster and get better performance. If you have a lot of data and start from random initialization, you can reach the same performance, but it will take a lot longer.

### Feature Extraction
Feature Extraction is using ==pre-trained frozen== layers as a backbone. These layers will extract features from the data for you which you can then pass to a couple layers that you do train.
<!--SR:!2024-06-07,621,330-->

You can train a CNN on ImageNet and then remove the last fully connected layer (you don't care what the scores are for ImageNet). You can then freeze the CNN layers and use these as a feature extractor. You can then just use the extracted features as a feature vector that represents the images.

![[AI-Notes/Representation Learning/transfer-learning-srcs/Screen_Shot.png]]

Models that work better on ImageNet tend to work better for transfer learning to every other problem.

This ends up being a non-linear transform before a linear layer. This transforms the input space so that it becomes linearly seperable. However, this is a learned feature transform (unlike HOG or SIFT).

### Fine-tuning
This is more powerful than feature-extraction and can be used if you have a lot of data. **Fine-tuning** will keep training the weights of the extracted layers instead of freezing the weights. You initialize the new layers with random initialization, but initialize the other layers with the pre-trained weights. You then train the entire model.

**Tricks to get fine-tuning to work**:
- You sometimes want to train the new layers with the frozen weights and get the new layers to converge. Then you can unfreeze the weights and train on all layers.
- Lower the learning rate to approximately $1/10$ of the LR used in original training. This helps prevent destroying the weights that were pre-trained.
- Sometimes freeze lower layers to save computation or save memory (don't need to save activations for back-prop).
- Train with BatchNorm in "test" mode

**Fine-tuning vs. feature extraction**:
- Requires more data
- Is more computationally expensive
- Can give higher accuracies

![[deciding-how-many-layers-to-fine-tune.png]]

Deciding how many layers to fine-tune:
- The latter layers of the model are more specific to ImageNet and the earlier layers are more general feature extractors
- If your model is very similar to ImageNet and you have very little data, use a linear classifier on the top layer and leave everything else alone
- If your model is very similar to ImageNet and you have lots of data, you can adjust a couple of layers.
- If your model is very different from ImageNet and you have a lot of data, then you can fine tune a larger number of layers
- If your model is very different from ImageNet and you don't have a lot of data, you are in a bad situation. You can try a linear classifier from different stages, but it isn't ideal.

> [!note]
> Most papers use some sort of transfer learning these days. It has become the norm, not the exception.

However, you can also get away without transfer learning, but it will take a lot more training and a lot of data.

Models that perform better on [[Pretext Tasks]] (like ImageNet) usually perform better when transferred to downstream tasks (the task you actually care about). This is part of why so many people focus on creating models that do well on benchmarks like ImageNet since it likely means the model will do better on other tasks when transferred. 

# Creating models for transfer learning
You need to train a model on one task so that you can re-use for other tasks. If we have a model try to do image classification, it requires human labelers annotating a lot of images. Instead, we want to be able to train a model on a large amount of data without supervision using [[Self-supervised Learning|self-supervised learning]]. 

![[AI-Notes/Representation Learning/transfer-learning-srcs/Screen_Shot 2.png]]

One example of this was a paper Professor Andrew Owens worked on that **correlated sound and the image they came from**. You can record videos and then the model is trained to predict the sound of the video based on the visual features. This model will hopefully learn valuable features that can be re-used.

You don't really care how good the predicted audio is. You just want the features learned to be valuable.

![[AI-Notes/Representation Learning/transfer-learning-srcs/Screen_Shot 3.png]]

Another way you could try to train a self-supervised model is trying to **predict color maps based on the original grey-scale image.**

### Autoencoders + Contrastive Learning
You could also use autoencoders or contrastive learning to try to get a good feature representation of an image.
[[Autoencoders]]
[[Contrastive Learning]]