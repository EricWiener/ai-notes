# C3D: VGG of 3D CNNs

Tags: EECS 498, Model

C3D is the equivalent of VGG for 3D CNNs. It is a 3D CNN that uses all 3x3x3 convolutions and 2x2x2 pooling (except Pool1 which is 1x2x2 - this only does pooling in space and not time).

![[AI-Notes/Video/c3d-vgg-of-3d-cnns-srcs/Screen_Shot.png]]

The model is pre-trained on Sports-1M and many people use this as a video feature extractor. This was popular because the authors released the weights for this model. For people who couldn't afford to train their own model, this could be used.

**Problem:** the 3x3x3 conv is very computationally expensive. 

- AlexNet: 0.7 GFLOP
- VGG-16: 13.6 GFLOP
- C3D: 39.5 GFLOP (2.9x VGG)

**Performance:** the model was able to get a better accuracy for the Sports-1M.

![[AI-Notes/Video/c3d-vgg-of-3d-cnns-srcs/Screen_Shot 1.png]]