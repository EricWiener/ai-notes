---
tags: [flashcards, eecs498-dl4cv]
source:
summary: represent the pose of a human by locating a set of keypoints.
---

![[AI-Notes/Image Segmentation/image-segmentation-srcs/Screen_Shot 21.png]]

You might want more fine-grained details about an object in an image. For instance, you might want to know where the eyes, ears, and nose are for a person in an image.

You can define a set of **keypoints**. You then want to predict where the keypoints are. 

# Keypoint Estimation with [[Mask R-CNN]]
![[screenshot-2022-03-27_11-34-26.png]]

Keypoint estimation can be done by attaching an additional head on-top of [[Instance Segmentation]]. Instead of a segmentation mask, you will predict a keypoint mask. You will have a different mask for each of your keypoints. The ground-truth has one pixel turned on per pixel.