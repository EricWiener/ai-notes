# Tesla Vision
- Vision processes information from the 8 cameras into the vector space (3D representation of everything you need for driving: lines, curbs, cars, traffic lights, etc).

### Tesla 4 years ago
- Tesla uses RegNet (can trade off latency and accuracy). You get features at different resolutions and scales.
- Process data with pyramid networks (BiFPN). Related paper: EfficientDet
- Have a one-stage YOLO-like model to detect objects

### Tesla HydraNets (several years ago)
![[screenshot-2022-03-12_10-01-01.png]]
- These use a shared backbone
- Each task in de-coupled so you can work on them in isolation.
- You don't need to re-validate other features if you fine-tune one specific feature
- Cache the features from the output of the back-bone so you don't need to propagate through the entire back bone in order to fine-tune. You can just go from the output features.

**Training workflow**
- Will do end-to-end training to train everything together
- Cache features at the multi-scale feature level
- Fine-tune off the cached features
- Repeat.

**Weaknesses**:
- You can't just drive on image space predictions
- They tried to create an occupancy grid from the features predicted on the individual images.
- Tuning the occupancy tracker was very complicated. You want this trained.
- You don't want to make predictions in image space - you want to make predictions in vector space.
![[screenshot-2022-03-12_10-05-25.png]]
- You end up with a bad representation in the vector space when trying to do per-camera features and then fuse.

### Multi-camera approach
![[screenshot-2022-03-12_10-07-21.png]]
- They process every image with a back-bone and then fuse them.
- Then they pass the fused images into the head.
- Problem 1: how do you create networks that perform the transformations.
- Problem 2: you need vector space datasets.

**Transformer**:
![[screenshot-2022-03-12_10-09-11.png]]
- They use a transformer with multi-head attention to figure out what pixels to look at in the image for the corresponding point in the vector space.
- You create a grid for your vectorized output space. You encode each position with a positional encoding. These then become the query vectors for your transformer. All the images are the key and value vectors. The queries, keys, and values then go into the attention layers.

**Problem: variations in camera calibrations**
- They use a rectification transform that goes right above the raw image
- It translates all the images into a virtual camera frame to handle each of the cameras in the different cars having different positions.

### Video Neural Net Architecture
![[screenshot-2022-03-12_10-15-08.png]]
- Currently they are using multiple cameras, but the cameras are all making predictions on individual time steps.
- They need to make use of video (ex: is a car parked, is a car traveling, is there an occuluded pedestrian, did we just drive over line markings, did we just pass a street sign, etc).
- They add a feature queue module to cache features over time.
- They also feed in kinematics (velocity, acceleration) to know how the car is moving.

**Feature queue**
![[screenshot-2022-03-12_10-16-12.png]]
- They store kinematic, camera, and positional encoding features.
- These are built up over time.
- These features are passed to the video module.
- When to push to queue? Ex: if you are stuck at an intersection, when do you want to push into the queue? If you push every $n$ ms, then you have a record over time. However, a space-based queue is also needed if you want to remember features from the road (ex. line markings and signs) that occurred before you arrived at the red light. They push every time the car travels a fixed distance and a fixed amount of time.

**Video Module**:
![[screenshot-2022-03-12_10-19-42.png]]
- They looked at 3D Conv, Transformer, Axial Transformers, RNNs
- They ended up going with a Spatial RNN Video Module
- They are driving on a 2D state. They use kinematics to integrate the position of the cars in the hidden features.
- Give the network the ability to select when it should read and write to memory.
- Having video also improved handling of occuluded vehicles and improved depth and velocity predictions.

### Current architecture
![[screenshot-2022-03-12_10-23-47.png]]
- Raw images come in from the bottom and go through a rectification layer to correct for camera calibration and put everything into a common virtual camera.
- The images then are passed through RegNets to process the cameras individually.
- The individual image features are fused via a BiFPN.
- A transformer module takes the fused features and re-represents it in the vector output space.
- They then have a feature queue in time or space
- The feature queue gets processed by a video module (ex. Spatial RNN)
- The video module features then goes into the branching structure of the HydraNet with trunks and heads for all the different tasks.

**Remaining problem**:
- Predicting in the [[Rasterization|rasterized]] vector space is expensive.
- Possible solution: use graphs?

# Planning and Control

# Data Labelling
- 