---
Files: 498_FA2019_lecture17.pdf
tags: [eecs498-dl4cv]
url: https://www.youtube.com/watch?v=S1_nCdLUQQ8&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r
---
![[AI-Notes/3D Object Detection/3D Vision/Screen_Shot.png]]

# Depth Map

![[AI-Notes/3D Object Detection/3D Vision/Screen_Shot 1.png]]

For each pixel, a depth map gives the distance from the camera to the object in the world at that pixel. An RGB image has dimensions: $3 \times H \times W$. A depth map has dimensions $H \times W$where each value is the depth in meters.

An RGBD image is an RGB image with a fourth channel for the depth. They are also called 2.5D (vs. 3D) images because they can't capture the structure of occluded objects (can only represent visible portions of the image).

This type of data can be recorded directly by some types of 3D sensors: Microsoft Kinect, Face ID, ZED, etc.

### Predicting Depth Maps

We can take an RGB image and try to predict a depth map for it. We can do this using a fully convolutional network to make one prediction per pixel. You can compare the predicted depth with the ground-truth depth and try to optimize this.

However, using just a single image, we have **scale/depth ambiguity.** Small objects that are close look the same size as large objects that are further away. 

![[Scale Depth Ambiguity]]

Scale/Depth Ambiguity

You need to use a **scale invariant loss** instead of just something like L2 distance to allow your model to be off by a scale factor.

# Surface Normals

![[AI-Notes/3D Object Detection/3D Vision/Screen_Shot 3.png]]

For each pixel, **surface normals** give a vector giving the normal vector to the object in the world for that pixel. This means it tells you what direction surfaces lie along.

If you predicted the surface normals, you would want to minimize the angle between the predicted normal vector and the normal vector in the ground truth.

You can train one network that will receive one RGB image and output segmented objects, depth predictions, and surface normal predictions in a single pass.

# Voxel Grid

![[AI-Notes/3D Object Detection/3D Vision/Screen_Shot 4.png]]

You represent the 3D world as a 3D grid. For each point in the grid, you either turn it on or off depending on if its occupied. It has shape $V \times V \times V$. It's like a Minecraft representation of the world.

It is nice because it is conceptually simple, but it also needs high spatial resolution to capture fine structures and scaling to high resolutions is nontrivial.

### Predicting class labels from voxel grids

Processing Voxel Grids is fairly simple. You can use a 3D convolution neural network, which is similar to a typical CNN, but each layer is a 3D convolution operation. You would train with classification loss.

![[AI-Notes/3D Object Detection/3D Vision/Screen_Shot 5.png]]

The 3D CNN would receive an input of size $1 \times 30 \times 30 \times 30$. The first dimension is the feature dimension and just tells us whether this voxel is occupied or not. The last three dimensions are for $x, y, z$. The latter layers can have more features, so this is why the second layer has an activation with dimensions $48 \times 13 \times 13 \times 13$. 

> [!note]
> This is pretty inefficient. A cubic voxel grid of size 100 will have 1,000,000 voxels.
> 

It is used quite a lot in practice.

### Predicting a voxel grid from an image with 3D CNN

If we receive a 2D image, we might want to predict a 3D voxel grid. We can use a 2D CNN to encode the image. Then, you can process these features with a fully connected layer. The outputs of the fully connected layer can then be converted into a three-dimensional output, which can be fed to a 3D CNN for predictions. You could train with per-voxel cross-entropy loss. 

![[AI-Notes/3D Object Detection/3D Vision/Screen_Shot 6.png]]

This is extremely computationally expensive because it involves 3D convolutions.

### Predicting a voxel grid from an image with 2D CNN

This is sometimes called a **voxel tube**. You use a 2D image with three channels (red, green, and blue). At the very end of the network, you treat the channels as if they were part of the spatial dimensions and your three dimensional tensor goes from $C \times H \times W$ to a voxel grid of size $V \times V \times V$. You again train with per-voxel cross-entropy loss.

![[AI-Notes/3D Object Detection/3D Vision/Screen_Shot 7.png]]

This is more efficient than using 3D convolutions. However, you lose translational invariance in the $z$ direction. In a normal 2D CNN, you apply the convolution throughout the image, so it doesn't matter if your cat is at the top-left or bottom-right of the image, since the convolution is applied over all the $x, y$. 

In a 3D CNN, you have a similar concept, but the filter is also applied over the $z$ direction in addition to the $x, y$ dimensions, so you can change the z-location and still get the same prediction. However, a 2D CNN doesn't have this property, so if you move an object in the z-direction, it won't be treated the same. 

### Scaling Voxels

You can try to save memory for a voxel grid by using tricks to avoid having to specify every value in the grid. This can be done using **oct-trees** or **nested shape layers.**

![[AI-Notes/3D Object Detection/3D Vision/Screen_Shot 8.png]]

![[AI-Notes/3D Object Detection/3D Vision/Screen_Shot 9.png]]

# Implicit Surface

![[AI-Notes/3D Object Detection/3D Vision/Screen_Shot 10.png]]

You want to represent the 3D shape as a function. You want to learn a function that receives some coordinate in 3D space and it outputs a prediction of whether that position is occupied or not occupied by an object: $o: \mathbb{R}^3 \rightarrow \{0, 1\}$.

We use a mathematical function to try to implicitly represent where the object is. It should be able to tell us whether points are inside or outside the object.

In the example above, the color of the implicit function tells us what the prediction would have been if we were too have evaluated it at that point in space (blue is 0 and red is 1). The surface is set to be $1 \over 2$.

This is also sometimes called the **signed distance function (SDF).** They are pretty equivalent for most purposes.

We will learn this function as a Neural Network. Receives a 3D coordinate and outputs whether it is inside or outside the shape. You can train on a sample of data points from your 3D shape. If you want to extract an exact shape, you can keep sampling from the function to extract the boundary of the real shape.

# Point Cloud

![[AI-Notes/3D Object Detection/3D Vision/Screen_Shot 11.png]]

We represent a 3D shape as a set of points in 3D space that will cover the surface of the shape we want to represent. You can see the airplane represented as many points on its surface.

They are better at representing fine 3D shapes than Voxel Grids because you can vary the density of the point cloud at different points in space (more points where the shape is more complex). With the Voxel Grid, all the points would need to be more dense, but you can only increase the density for some points with a point cloud.

You need some post-processing if you want an actual 3D shape to visualize. Each of the points are infinitely small, so there is no way to visualize them. You need to inflate them to a ball (as seen in the above example). It needs post-processing to visualize it. Additionally, you need new architecture and losses.

Useful to work with in Neural Networks and is commonly used in self-driving cars. Point cloud info is collected by Lidar.

> [!note]
> Most 3D scanning devices produce point cloud data
> 

## Predicting object class based on point cloud

[[PointNet]]

## Predicting point cloud given a 2D image

![[AI-Notes/3D Object Detection/3D Vision/Screen_Shot 12.png]]

In order to train a model that predicts point clouds, we now need a loss function that can compare two sets of data (since point clouds are unordered). This function needs to be differentiable so we can backpropagate through it.

### **Chamfer Distance:**

![[AI-Notes/3D Object Detection/3D Vision/Screen_Shot 13.png]]

This can be done using the **chamfer distance**. It is the sum of the L2 distance to each point's nearest neighbor in the other set.

The first term looks at every blue-point and finds the closest orange orange point. It adds up all these distances.

![[AI-Notes/3D Object Detection/3D Vision/Screen_Shot 14.png]]

![[AI-Notes/3D Object Detection/3D Vision/Screen_Shot 15.png]]

The second term looks at every orange-point and finds the closest blue point. It adds up all these distances.

![[AI-Notes/3D Object Detection/3D Vision/Screen_Shot 16.png]]

![[AI-Notes/3D Object Detection/3D Vision/Screen_Shot 17.png]]

> [!note]
> The only way to get the chamfer distance to be zero is if the two point clouds are exactly the same. However, the order of the points does not matter, since we are just using the nearest neighbor.
> 

# Triangle Mesh

![[AI-Notes/3D Object Detection/3D Vision/Screen_Shot 18.png]]

The shape is represented as a set of vertices in 3D space (like point cloud). However, it also has a set of triangular faces (triangles with vertices on the point cloud). This will represent the shape as a set of triangles.

It is commonly used for computer graphics. It is adaptive like point clouds because you can have bigger or smaller points in the model (smaller triangles for more exact details). You can also explicitly represent 3D shapes. Additionally, you can add RGB color, texture, etc. to each vertex, so you can extrapolate this data over the triangle mesh without having to store it for every point in the triangle (this is often how 3D textures are rendered).

It is nontrivial to process these with neural networks.

## Predicting Meshes from RGB image: Pixel2Mesh

![[AI-Notes/3D Object Detection/3D Vision/Screen_Shot 19.png]]

This model took a single RGB image of an object as input and outputted a triangle mesh for the object.

### **Idea #1: Iterative mesh refinement**

![[AI-Notes/3D Object Detection/3D Vision/Screen_Shot 20.png]]

You start from an initial ellipsoid mesh. The network will then continuously refine the positions of the vertices. You will keep refining the mesh to make it match the input image as much as possible.

### **Idea #2: Graph Convolution**

This is a similar to 2D and 3D convolutions, but now it is extended to arbitrary graph structured data. The input to the layer will be a graph with feature vectors attached to every node in the graph.

The output of the layer will have a new feature vector for every node in the graph. The new feature vector should depend on features of the neighboring vertices from the input graph.

![[AI-Notes/3D Object Detection/3D Vision/Screen_Shot 21.png]]

![[AI-Notes/3D Object Detection/3D Vision/Screen_Shot 22.png]]

- Vertex $v_i$ has feature $f_i$
- New feature $f'_i$ for vertex $v_i$ will depend on features of neighboring vertices $N(i)$
- You use the same weights $W_0$ and $W_1$ to compute all the outputs

You can convolve over the input graph no matter what the graph's topology. We can now process the mesh using the graph convolution. We will attach a feature vector to every vertex in the mesh. After every convolution, it will update these feature vectors depending on their neighbors.

### Idea #3: Vertex-Aligned Features

For every vertex in the mesh, we want to get a feature vector from the image that represents the visual appearance of the image at that spatial position of the image. 

We will use a 2D CNN to extract image features from the image. If we understand the camera intrinsics, we can then project the vertices of the mesh onto the image plane. 

We can then use bilinear interpolation to sample from these image features. We now have a feature vector for each vertex in the mesh (similar to RoI align in Mask R-CNN).

### Idea #4: Loss Function

We need a loss function to compare 3D triangle meshes. However, we can represent the same 3D shape using different meshes. For instance, a square can be represented as two triangles or four triangles.

![[The same shape with different meshes]]

The same shape with different meshes

We want our loss function to be invariant to how we represent the shape. Our meshes will be converted into point clouds by sampling points on the surface of the mesh. We can then use the **chamfer loss function** to compare the two point clouds.

![[AI-Notes/3D Object Detection/3D Vision/Screen_Shot 24.png]]

We can pre-compute the sampled points for the ground truth, but we need to compute the point cloud representation for the prediction online. We also need to be able to backpropagate through this sampling.