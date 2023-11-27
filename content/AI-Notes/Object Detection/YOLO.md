---
tags: [flashcards, eecs498-dl4cv]
source:
summary:
---
![[AI-Notes/Concepts/object-detection-srcs/Screen_Shot 12.png|400]]

You start with the same convolutional feature map as Fast R-CNN and then you divide it into a grid. Instead of proposing regions (saying whether regions are objects/not objects), you give probabilities each cell belongs to a particular class. Then you regress the bounding box for each cell. **The bounding box regression now is aware of the class prediction (previously it wasn't).**

The bounding boxes with the highest confidence are given as the final detection.

### YOLO Detector
1. Take convolutional feature maps at $7 \times 7$ resolution (the CNN features will be of size 7 by 7). 
2. Predict, at each location, a score for each class and 2 bounding boxes with confidence. If you have 20 classes, you have $7 \times 7 \times 30$ bounding boxes. The 30 comes from $30 = 20 + 2 \times (4 + 1)$. 
    - For each of the cells in the $7 \times 7$ grid, you output a probability for each of the $20$ classes.
    - Additionally, you output 2 bounding boxes with a $x, y, w, h$ and a confidence score $2 \times (4 + 1)$
    - This is $30$ values given per cell.

It is seven times faster than Faster-RNN, but less accurate. 

# Two-stage vs. one-stage detectors
- Two stage methods like [[Fast R-CNN]] first generate some region proposals and then decide whether each region contains an object and do bounding box regression to get a refined box.
- One-stage methods like [[YOLO]] directly predict the offset of real boxes relative to anchor boxes.