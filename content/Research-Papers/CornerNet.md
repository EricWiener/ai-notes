---
summary: 
tags:
  - eecs498-dl4cv
---

Both the two stage (R-CNN's) and one stage (YOLO) relied on anchor boxes. CornerNet (a paper published at Michigan) tried to get rid of the need for anchor boxes.  

![[AI-Notes/Concepts/object-detection-srcs/Screen_Shot 16.png]]

They represent bounding boxes of pairs of upper-left corners and lower-right corners. Then, for each pixel in the image, they generate a probability of how likely it is to be an upper-left corner or a lower-right corner for a bounding box. The pixels also have a corresponding embedding vector which describes the object it is a bounding box for.

You then get a heat map of upper-left corners and lower-right corners. You can try to match these up with one another by seeing if they have similar embedding vectors.