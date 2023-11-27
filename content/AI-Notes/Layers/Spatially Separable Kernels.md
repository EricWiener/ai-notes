# Spatially Separable Kernels

[Great article](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728)

- A spatial separable convolution simply divides a kernel into two, smaller kernels. The most common case would be to divide a 3x3 kernel into a 3x1 and 1x3 kernel.

$$
\left[\begin{array}{ccc}3 & 6 & 9 \\4 & 8 & 12 \\5 & 10 & 15\end{array}\right]=\left[\begin{array}{l}3 \\4 \\5\end{array}\right] \times\left[\begin{array}{lll}1 & 2 & 3\end{array}\right]
$$

Now, instead of doing one convolution with 9 multiplications, we do two convolutions with 3 multiplications each (6 in total) to achieve the same effect. With less multiplications, computational complexity goes down, and the network is able to run faster.

![[AI-Notes/Video/Untitled]]

> [!note]
> Separable kernels divide the kernel spatially (along the height/width dimension) in order to reduce the number of operations needed while still getting the same final result.
> 

### Sobel Kernel

$$
\left[\begin{array}{lll}-1 & 0 & 1 \\-2 & 0 & 2 \\-1 & 0 & 1\end{array}\right]=\left[\begin{array}{l}1 \\2 \\1\end{array}\right] \times\left[\begin{array}{lll}-1 & 0 & 1\end{array}\right]
$$

- The Sobel Kernel is used to detect edges in traditional computer vision and is a famous example of a separable kernel.

### Only a small number of kernels are linearly separable

- If a kernel is linearly separable or not depends on the values within the kernel - not on the shape of the kernel.
- Most 3x3 kernels aren’t linearly separable, so it is uncommon to be able to split them the same way you could for the special case of the Sobel Kernel.

> [!note]
> Since few kernels are linearly separable, if you made all your convolutional layers the size of their separable parts (i.e. turned all 3x3 conv → 3x1 and 1x3 conv), you would only be able to learn a **subset** of all kernels the original network could have learned.
> 