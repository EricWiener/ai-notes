---
tags: [flashcards, eecs498-dl4cv]
source: https://arxiv.org/abs/1801.00868
summary: Panoptic segmentation labels all pixels in the image (both things and stuff). However, it also separates things into separate instances.
---

> [!NOTE] The goal of panoptic segmentation is to predict a set of non-overlapping masks along with their corresponding class labels.

 ![[AI-Notes/Image Segmentation/image-segmentation-srcs/Screen_Shot 19.png|300]]

 ![[AI-Notes/Image Segmentation/image-segmentation-srcs/Screen_Shot 20.png|300]]
 
 Instance segmentation separates instances, but only works for things. Semantic segmentation identifies both things and stuff, but doesn't separate them. Panoptic segmentation labels ==all pixels in the image (both things and stuff). However, it also separates things into separate instances==.
<!--SR:!2024-07-15,443,310-->

**Helpful description from [[Panoptic-DeepLab]]**:
The goal of panoptic segmentation is to assign a unique value, encoding both semantic label and instance id, to every pixel in an image.

It requires identifying the class and extent of each individual ‘thing’ in the image, and labelling all pixels that belong to each ‘stuff’ class.

**Alternative phrasing from [[UPSNet]]**:
In panoptic segmentation, countable objects (those that map to instance segmentation tasks well) are called things whereas amorphous and uncountable regions (those that map to se- mantic segmentation tasks better) are called stuff. For any pixel, if it belongs to stuff, the goal of a panoptic segmentation system is simply to predict its class label within the stuff classes. Otherwise the system needs to decide which instance it belongs to as well as which thing class it belongs to.

# Panoptic Segmentation Paper
Computer vision research has previously handled [[Instance Segmentation]] and [[Semantic Segmentation]] seperately. The schism between semantic and instance segmentation has led to a parallel rift in the methods for these tasks. Stuff classifiers are usually built on fully convolutional nets with [[Dilated Convolution]] while object detectors often use object proposals and are region-based.

This paper tries to bridge the two fields. The definition of "panoptic" is "including everything visible in one view."

Unlike semantic segmentation, it requires differentiating individual object instances; this poses a challenge for fully convolutional nets. Unlike instance segmentation, **object segments must be non-overlapping**; this presents a challenge for region-based methods that op- erate on each object independently. Generating coherent image segmentations that resolve inconsistencies between stuff and things is an important step toward real-world uses.

**Task**: each pixel of an image must be assigned a semantic label and an instance id. Pixels with the same label and id belong to the same object; for stuff labels the instance id is ignored.