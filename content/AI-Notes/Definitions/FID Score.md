---
tags: [flashcards]
source: https://wandb.ai/ayush-thakur/gan-evaluation/reports/How-to-Evaluate-GANs-using-Frechet-Inception-Distance-FID---Vmlldzo0MTAxOTI
summary:
---

The Frechet Inception Distance score, or FID for short, is a metric that calculates the distance between feature vectors calculated for ==real and generated== images. [Source](https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/) A lower score is better.
<!--SR:!2024-09-06,314,310-->

Two properties we want for our evaluation metric are:
- Fidelity: We want our GAN to generate high quality images.
- Diversity: Our GAN should generate images that are inherent in the training dataset.

Two approaches to compare images that are widely used in computer vision are:
- Pixel Distance: This is a naive distance measure where we subtract two images' pixel values. However, this is not a reliable metric.
- Feature Distance: We use a pre-trained image classification model and use the activation of an intermediate layer. This vector is the high-level representation of the image. Computing a distance metric with such representation gives a stable and reliable metric.

The FID score is calculated by using the activations (the last pooling layer prior to the output classification of images) from an Inception V3 pre-trained model on ImageNet.