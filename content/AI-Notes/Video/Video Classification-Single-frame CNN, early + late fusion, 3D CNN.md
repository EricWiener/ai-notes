# Video Classification: Single-frame CNN, early + late fusion, 3D CNN

Files: 498_FA2019_lecture18.pdf, lec14-action.pdf
Tags: EECS 442, EECS 498

Like we did with 3D object detection, a video can be represented as a 4-dimensional tensor. We can either have $T \times 3 \times H \times W$ or $3 \times T \times H \times W$. Both can be used.

# Video Classification

As a motivating example of when we might work with videos, we can use video classification. With images, we wanted to classify objects (dog, cat, etc.). With videos, we want to classify actions (running, swimming, etc.). The majority of these actions are actions people can do. This is usually what people care about.

### Problem: videos are big

A standard video is shot at ~30 frames per second (fps). If we use uncompressed video, where we have a value for the red, green, and blue channels for each frame, we need 3 bytes per pixel. 

- SD (640 x 480): ~1.5 GB per minute
- HD (1920 x 1080): ~10 GB per minute

> [!note]
> You can't fit long sequences of videos into your GPU memory.
> 

The solution to this is to train on **short clips**: low frames per second and low spatial resolution (e.g. T = 16, H=W=112).

![[AI-Notes/Video/video-classification-single-frame-cnn-early+late-fusion-3d-cnn-srcs/Screen_Shot.png]]

During training, you can train on short clips of the video with reduced frame rate (sub-sample the clip). During testing, you can run the model on multiple clips and average the predictions.

# Video Classification: Single-Frame CNN

This is a simple approach, but it works well. You train a normal 2D CNN to classify video frames independently. You then average the predicted probabilities at test-time. **This will often work good enough for many applications. Always try this first.**

![[AI-Notes/Video/video-classification-single-frame-cnn-early+late-fusion-3d-cnn-srcs/Screen_Shot 1.png]]

One issue with this is you can permute the frames (shuffle them) and you'll  get the same result. You can't analyze motion.

# Video Classification: Late Fusion

This modifies the single-frame CNN approach. Instead of just averaging up all the class probabilities provided by the CNN, we instead just use the CNN to extract features (it no longer gives predictions). We then flatten all these features and concatenate them for each time-step. You then pass the flattened features to a fully connected network, which gives you you class scores.

This is called late fusion because you fuse the temporal information at a very late phase of the pipeline.

![[AI-Notes/Video/video-classification-single-frame-cnn-early+late-fusion-3d-cnn-srcs/Screen_Shot 2.png]]

### Video Classification: Late Fusion (with pooling)

We can replace the idea of flattening with using global average pooling. The pooling will be performed over all the spatial and over all the temporal dimensions. It will leave us with just a $D$ dimensional vector (collapsing $T, H', W'$). We can then pass this to a linear layer.

![[AI-Notes/Video/video-classification-single-frame-cnn-early+late-fusion-3d-cnn-srcs/Screen_Shot 3.png]]

**Problem:** the issue with late fusion is that it can't detect very low-level movements between video frames. For instance, in the video of the runner above, it will be hard for the small movements of the foot to be picked up. You summarize all the information about each frame in a single vector. This is a short-coming of late fusion.

# Video Classification: early fusion

With late fusion, we fuse all the temporal information at the end of the pipeline. With early fusion, we fuse all the information at the beginning of the pipeline. You can reshape the input from $T \times 3 \times H \times W$ to be $3T \times H \times W$. You can interpret the temporal information as just additional channels. You can now use a regular 2D CNN on this reshaped tensor.

![[AI-Notes/Video/video-classification-single-frame-cnn-early+late-fusion-3d-cnn-srcs/Screen_Shot 4.png]]

The first 2D convolution is responsible for collapsing all the temporal information from $3T \times H \times W$ to $D \times H \times W$.  This 2D convolution can have weights that are specific to temporal information or color channel information. However, after this layer, all the information will be mixed together.

### Problem:

One layer of temporal processing may not be enough. 

Additionally, we have the issue that there is no **temporal shift-invariance.** For instance, if you wanted to have a filter learn to recognize a shift from blue to orange, it would need to learn separate filters for the same motion at different times in the clips.

![[AI-Notes/Video/video-classification-single-frame-cnn-early+late-fusion-3d-cnn-srcs/Screen_Shot 5.png]]

You need different filters for different positions in time.

### Temporal Filtering Visualization

The idea of merging the temporal dimension and channel dimension is also known as temporal filtering, since you are applying a filter over time as well. The following is a visualization of what this could look like:

![[AI-Notes/Video/video-classification-single-frame-cnn-early+late-fusion-3d-cnn-srcs/Screen_Shot 6.png]]

We can create a cube that has both spatial and temporal dimensions. On the right is a single slice where we show a row of the images for all time steps.

We can see the lines are motion and the static background is relatively in place (a bit blurry because of camera motion).

![[AI-Notes/Video/video-classification-single-frame-cnn-early+late-fusion-3d-cnn-srcs/Screen_Shot 7.png]]

We can also take a vertical snapshot in time and we see that now we can see people. This is because they walked through the vertical part of the image, so they appear in multiple time steps.

### Temporal Median Filtering

You can apply hand-crafted filters to the video in its new form of $(3T \times H \times W$). For instance, you can apply a median filter to every pixel in the space-time volume. You can do this to erase people.

![[AI-Notes/Video/video-classification-single-frame-cnn-early+late-fusion-3d-cnn-srcs/Screen_Shot 8.png]]

You can also subtract the background image (**background subtraction)** and you can use it to find the non-background objects to detect people.

# 3D CNN: Space-time Convolution

Instead of using handcrafted features to analyze video, you can apply a **3D space-time CNN** that convolves over both the spatial and temporal dimensions. You use 3D versions of both convolutions and pooling to fuse temporal information over the course of many layers.

![[AI-Notes/Video/video-classification-single-frame-cnn-early+late-fusion-3d-cnn-srcs/Screen_Shot 9.png]]

Now each feature (channel) has 3 dimensions instead of two.

3D CNN's are sometimes called a **slow fusion** approach because they fuse the temporal and spatial information slowly over time.

### Temporal shift-invariance

3D convolutions now have **temporal shift-invariance** since each filter slides over all of time.

![[AI-Notes/Video/video-classification-single-frame-cnn-early+late-fusion-3d-cnn-srcs/Screen_Shot 10.png]]

![[AI-Notes/Video/video-classification-single-frame-cnn-early+late-fusion-3d-cnn-srcs/Screen_Shot 11.png]]

Previously, with the 2D convolution, the filter extended over all the temporal dimension. When you convolved it with the input, you would just move it across the spatial dimensions ($H, W$) and you would end up with a 2D output. 

However, a 3D convolution only extends over a small temporal amount, so it will be moved across $T, H, W$. You will now get a 3D output because it will look at multiple different times. This has the benefit that the filter can detect features no matter what time they occur in the sequence.