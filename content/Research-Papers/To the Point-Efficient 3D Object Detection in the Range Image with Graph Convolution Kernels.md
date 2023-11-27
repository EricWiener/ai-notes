# To the Point: Efficient 3D Object Detection in the Range Image with Graph Convolution Kernels

PDF: https://drive.google.com/open?id=1F7UnOXEAchPBimdQEJ0Z5YUxnSQtQYSb&authuser=ecwiener%40umich.edu&usp=drive_fs
Reviewed: No
summary: Presented in Zoox ML presentation
url: https://arxiv.org/abs/2106.13381

- Problem with a 3D grid of points is that you face computational + memory complexities. Can alleviate by making voxels larger, but then you get discretization affects.
- Range view (aka perspective point cloud) works better than birds eye view for tall and thin objects.

![[/Screenshot 2022-02-17 at 19.07.23@2x.png]]

- On the left is the top-down view. On the right is a range view. Each row is discretized elevation angle.
- 2D kernels don’t work well for range view.
- They propose new kernels to replace 2D kernels.
    - Dense kernel
    - Other three are like graph neural network kernels
    - 

### Questions

- What is a range image view?