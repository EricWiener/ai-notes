
---
summary: Computational graphs allow us to switch from thinking of the model as a large algebraic equations to small nodes that allow us to just need to calculate downstream gradients by multiplying local gradients by upstream gradients.
tags: [eecs498-dl4cv]
---
[[598_FA2020_lecture06.pdf.pdf]]
[[gradient-notes.pdf.pdf]]

This lecture covers how to calculate the gradients for arbitrarily complex Neural Networks. You may attempt to derive the gradients on paper. **However, this is very tedious, error prone, and needs to be re-calculated whenever any aspect of your model changes.** It's much better to have a modular approach where you can swap around different parts of the architecture.

Our goal is to be able to calculate $\frac{\partial L}{\partial W_{i}}, \frac{\partial L}{\partial b_{i}}$ for all weights and biases in the model. This tells us the derivative of the loss with respect to a particular weight or bias. We can then use this gradient to update our weight and bias vectors.

## Convex Functions

A function is convex if it is shaped like a high dimensional bowl. Convex functions are **easy to optimize**: can derive theoretical guarantees about **converging to global minimum**.

Linear classifiers optimize a **convex function** (SVM and Softmax). This means linear classifiers (under some assumptions) can converge to a global minimum.

However, Neural Networks don't have the same nice theoretical guarantees because they aren't guaranteed to be convex. They need **nonconvex optimization.** In practice, they seem to converge nicely anyways.

# Computational Graph

We use computational graphs to help us solve for the gradients. It is a directed graph that helps us represent the computation we perform inside the model.

![[AI-Notes/Concepts/backpropagation-srcs/Screen_Shot.png]]

In this example of the computational graph, the learnable weights $W$ and the input $x$ are put  into the graph and combined with a matrix multiply, represented by the grey `*`. The output scores are passed through hinge loss and then added with an `R` term (regularization on $W$) to calculate the final scalar loss.

### Example

Backpropagation for a simple example: $f(x, y, z) = (x + y)z$

We want to evaluate the gradient at a particular instance where:

 $x = -2, y = 5, z = -4$.

![[AI-Notes/Concepts/backpropagation-srcs/Screen_Shot 1.png]]

1. **Forward pass**: compute the outputs

1. $q = x + y$
2. $f = qz$

![[AI-Notes/Concepts/backpropagation-srcs/Screen_Shot 2.png]]

2. **Backward pass:** compute the gradient of the output with respect to each of the inputs (see how $f$ will change if we modify $x$, $y$, or $z$ a little bit). We want to compute the derivatives of the output with respect to the inputs $\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z}$. We work backwards from right to left since we always start with the output.
![[IMG_E73844E69245-1.jpeg.jpeg]]
- $\frac{\partial f}{\partial f}=1$ because the derivative of a value with respect to itself is always 1
- $\frac{\partial f}{\partial z} = q = 3$ because $f = qz$ so $\frac{\partial f}{\partial z} = q\cdot 1 = 3$
- $\frac{\partial f}{\partial q}=z=-4$ because $f = qz$, so $\frac{\partial f}{\partial q}=z\cdot 1=-4$

We can use the chain rule to calculate $\frac{\partial f}{\partial x}$ because it doesn't directly have a connection to $f$, we can do $\frac{\partial f}{\partial y} = \frac{\partial q}{\partial y}\cdot \frac{\partial f}{\partial q}$.

- $\frac{\partial q}{\partial y} = \frac{\partial q}{\partial y} (x + y) = 1$ (the derivative of $x + y$ with respect to $y$ is $0 + 1$)
- $\frac{\partial f}{\partial q} = \frac{\partial f}{\partial q} (qz) = -4$
- $\frac{\partial f}{\partial y} = 1 \cdot -4 = -4$

Note that you don't really care what $\frac{\partial f}{\partial x}$ is since you can't optimize your training examples. You only care what this is if it is not the left-most layer of the computation graph, since it will then become an upstream gradient (ex. multiple fully connected layers used sequentially).

## Downstream, Local, Upstream Gradients

![[AI-Notes/Concepts/backpropagation-srcs/Screen_Shot 3.png]]

The **downstream** gradient is the gradient we are currently trying to compute ($\frac{\partial f}{\partial y}$ in the above figure). The **local** gradient is the local affect of how much the value of $y$ affects the intermediate value of $q$. The **upstream** gradient is how much the output of the local gradient affects the rest of the graph.

![[AI-Notes/Concepts/backpropagation-srcs/Screen_Shot 4.png]]

Each piece of the graph doesn't need to care about the rest of the graph. When forward propagation occurs, the node $f$ will recieve inputs $x$ and $y$ and will give an output $z$. The network will keep performing operations until a very far away node calculates the final loss, $L$. 

Back propagation will begin and the gradients will keep being passed backwards. Eventually the node will receive the **upstream** gradient $\frac{\partial L}{\partial z}$. It can then compute the **local** gradients, which say how much the inputs of the node affect the output (these are $\frac{\partial z}{\partial x}$ and $\frac{\partial z}{\partial y}$ here). Finally, it can use the chain rule to compute the **downstream** gradient, which says how much the inputs of the node affect the final loss. The downstream gradients are then passed off to previous nodes as the upstream gradient and the **cycle repeats**.

## Modular Computations

[[backprop-walkthrough.pdf]]

You can choose how much you want to break down the computation and what to consider nodes. For instance, you could consider the entire sigmoid function to be a single node. This allows this node to be re-used in other networks and its gradient will already be calculated.
![[softmax-from-linear-classifier-graph.png]]
This shows computing the softmax of the outputs of a linear classifier.

The local gradient of sigmoid simplifies to $\frac{\partial}{\partial x}[\sigma(x)] = (1-\sigma(x)) \sigma(x)$. This means the gradient of the sigmoid node is equal to the output of the sigmoid * (1 - the output of the sigmoid). You can then calculate the downstream gradient in the above example with:

- downstream = (1 - sigmoid output)(sigmoid output) * (upstream gradient).
- 0.2 = (1 - 0.73)(0.73)(0.2)

## Patterns in Gradient Flow

The add operation will have the **same downstream gradients as upstream gradient**. This is because the gradient of a simple add operation ($x + y$) is 1.

![[add-gate.png]]

The copy gate's downstream gradient will be the **sum of its upstream gradients**. A copy gate just copies its input to two different outputs (useful if you want an input to be used in multiple places). 

![[AI-Notes/Concepts/backpropagation-srcs/Screen_Shot 7.png]]

It makes sense that it sums the upstream gradients because the same input is affecting multiple outputs. You might use the weights once to compute the output of a layer and once when computing the regularization.

The multiplication gate multiplies the upstream gradient by the **other** input to calculate the downstream gradient. You can see the multiplication gate will multiply the inputs together for forward-prop and also multiply again during back-prop. This can cause the gradients to become very large.

![[AI-Notes/Concepts/backpropagation-srcs/Screen_Shot 8.png]]

The **max** gate will only pass the upstream gradient to one downstream gradient (the other will be zero) because only one input gets passed to the output.

![[max-gate.png]]

The **reciprocal** gate will have a local gradient of $\frac{-1}{x^2}$. You then multiply this by the upstream gradient. 

![[reciprocal-gate.png]]

The **exponential** gate has a local gradient of $e^x$, so you just multiply the upstream gradient by the output of this node.

![[exponential-gate.png]]

The **log** gate has a local gradient of $\frac{1}{x}dx$, so you just multiply the upstream gradient by the reciprocal of the input to this node.

![[log-gate.png]]
![[summation-gate.png]]
The summation gate

The **summation gate** sums up the values during the forward pass. Just like an addition gate, it just passes the gradient backwards to all the inputs evenly. You also divide by $\frac{1}{N}$ because this is the local gradient (this could really be split into a summation and division node). ****

## Flat Backprop
![[flat-backprop.jpeg]]
You can explicitly write out all the steps of the back-propagation in the reverse order of the forward propagation. This works and is much easier than writing out the calculations manually, but it still isn't very modular. You end up with a 1-1 correspondence between lines of code in your forward pass and your backward pass.

## Backprop Implementation: Modular API
![[psuedo-code-for-computational-graph.png]]
Rough pseudo-code for a Computational Graph

You would have a computational graph that in the forward pass first topologically sorts the nodes so they are in the correct order and then calls `forward` on each node. In the backward pass it would reverse sort the nodes and call `backward`.

In PyTorch, You can use `PyTorch Autograd`  to implement your own forward and backward propagation functions for a certain node. 

![[AI-Notes/Concepts/backpropagation-srcs/Screen_Shot 13.png]]

- You can use `ctx_save_for_backward` to save the values `x` and `y` for when you do back-prop.
- In `backward`, you receive `ctx` and the upstream gradient, `grad_z`.

PyTorch has a huge library of forward/backward pairs of functions that automatically compute the gradients for you.

# Back-prop with vectors

So far we have only been using scalars for back-prop, but we also want to be able to use vectors.

## Re-cap: Vector Derivatives

**Regular Derivative:**

- $x \in \mathbb{R}, y \in \mathbb{R}$
- $\frac{\partial y}{\partial x} \in \mathbb{R}$
- If $x$ changes by a small amount, how much will $y$ change?

**Vector input, scalar output:**

- $x \in \mathbb{R}^N, y \in \mathbb{R}$
- $\frac{\partial y}{\partial x} \in \mathbb{R}^N$  $(\frac{\partial y}{\partial x})_n = \frac{\partial y}{\partial x_n}$
- The derivative is a **gradient.** The gradient is a vector that is the same size of the input. Each element tells you how much $y$ changes if the corresponding element of the input $x$ changes.

**Vector input, vector output:**

- $x \in \mathbb{R}^n, y \in \mathbb{R}^M$
- $\frac{\partial y}{\partial x} \in \mathbb{R}^{N \times M}$  $(\frac{\partial y}{\partial x})_{n, m} = \frac{\partial y_m}{\partial x_n}$
- The derivative is a **Jacobian.** It is a matrix that is $N \times M$ (input x output). For each element of the input and for each element of the output, how much does changing that element of the input affect that element of the output. The Jacobian is a matrix of partial derivatives.
![[jacobian-matrix-w-partials.png]]
Example showing the how the Jacobian matrix encapsulates the partial derivatives

## Backprop with Vectors

![[AI-Notes/Concepts/backpropagation-srcs/Screen_Shot 15.png]]

Now we are dealing with vectors. $x$ has dimensions $(D_x, 1)$, $y$ has dimensions $(D_y, 1)$ and $z$ has dimensions $(D_z, 1)$. The loss $L$ is a scalar.

Because we are now dealing with vector inputs and vector outputs, we need to adjust our approach to handle matrices.

- The loss, $L$, is still a scalar
- The upstream gradient, $\frac{\partial L}{\partial z}$ now has dimensions $(D_z, 1)$ - the same dimensions as the output of the node, $z$. This tells us how changing each element of the $z$ will affect the final loss, $L$. The upstream gradient is sometimes called the **error signal** since it fully encapsulates the error that the later layers tell us. Note: $\frac{\partial L}{\partial z}$ has the same shape as $z$ because the gradient of a scalar with respect to a vector always has the same shape as the vector.
- We calculate the local Jacobian matrices for each input which tells us how adjusting each element of our inputs will affect the output, $z$.
- We can then use a matrix-vector product to get the downstream gradients, which has the same dimensions as their corresponding input. For example, $\frac{\partial L}{\partial x} = \frac{\partial z}{\partial x} \frac{\partial L}{\partial z}$. $\frac{\partial z}{\partial x}$ has dimensions $(D_x, D_z)$ and $\frac{\partial L}{\partial z}$ has dimensions $(D_z, 1)$. Therefore, $\frac{\partial z}{\partial x} \frac{\partial L}{\partial z}$ has dimensions $(D_x, D_z) \times (D_z, 1)$ which is $(D_x, 1)$ - the same dimension as $x$.

### Example with ReLU

![[AI-Notes/Concepts/backpropagation-srcs/Screen_Shot 16.png]]

In the top of the image, you compute an element wise ReLU to set values of the input less than zero to 0. You then get your output $y$.

You eventually receive the upstream gradient $\frac{\partial L}{\partial y}$. We compute the local Jacobian matrix. Because ReLU is an element wise function, the Jacobian is a diagonal matrix because the ith entry of the input will only affect the ith entry of the output. The corresponding entry is 1 if the corresponding input entry was > 0 and 0 otherwise. Note: the upstream gradient is all non-zero even though the output $y$ containd 0s. This is possible because networks have bias terms, so having output at one hidden layer be 0 won’t necessarily kill your upstream gradient (but it will kill the downstream gradient).

We then compute the matrix-vector multiply between the local Jacobian and the upstream gradient to get the downstream vector that will be passed to other nodes as the upstream gradient.

> [!note]
> We notice here the Jacobian is sparse. This is usually the case in Deep Learning. Most of the functions have very sparse Jacobians. You will almost never explicitly form the Jacobian and explicitly perform the matrix-vector multiply.
> 

The big trick in back-prop is figuring out how to express the Jacobian vector multiplies in an implicit, efficient way. Below is how it can be done for ReLU, where you only pass on the gradients where the corresponding $x_i > 0$ (wasn't set to 0 during forward-prop).

Trying to use the full Jacobian will quickly make you run out of memory (for even medium sized networks).

![[AI-Notes/Concepts/backpropagation-srcs/Screen_Shot 17.png]]
Replacing the matrix-vector product for the ReLU.

Here we can set the downstream gradient to the corresponding entry in the upstream gradient if $x_i > 0$ and otherwise set the downstream gradient to 0. This allows us to avoid needing to form the Jacobian matrix in memory.

> [!note]
> The Jacobian is sparse. Never **explicitly** form Jacobian. Instead use **implicit** multiplication.
> 

## Backprop with Matrices (or Tensors):

![[AI-Notes/Concepts/backpropagation-srcs/Screen_Shot 18.png]]

The upstream gradient $\frac{\partial L}{\partial z}$ always has the same shape as $z$ because it tells you how much each element of $z$ affects $L$ (the loss - which is always a scalar). The Jacobians tell you how much each element of the input affects each element of the output. They are 4D matrices with dimensions $[\mathrm{D}_{\mathrm{x}} \times \mathrm{M}_{\mathrm{x}} \times \mathrm{D}_{\mathrm{z}} \times \mathrm{M}_{\mathrm{z}}]$. 

To calculate the downstream gradients, you perform a matrix-vector multiply. The local Jacobians are 4D matrices. The upstream gradients are vectors and you want the downstream gradients to be vectors as well (to match the shape of the input). Therefore, $\frac{\partial z}{\partial x} \frac{\partial L}{\partial z}$ is a matrix * a vector = a vector

This is very complicated and hard to think about when you have very high dimensional tensors.

You should look for tricks to avoid explicitly forming the Jacobian and find ways to calculate the downstream gradient from the upstream gradient by finding patterns.

We know it has to involve the input and we know it has to involve the upstream gradient, so we can look at how the shapes work out to find how to combine them.

![[Screenshot_2022-01-26_at_07.11.442x.png]]

We can implicitly calculate $\frac{\partial L}{\partial x}$ by multiplying the upstream gradient by the transpose of the weight matrix. We can implicitly calculate $\frac{\partial L}{\partial w}$ by multiplying $x^T$ by the upstream gradient. This is the only way the shapes work out.

## Reverse Mode Automatic Differentiation

This is done when we have a **vector input and a scalar output.**

![[AI-Notes/Concepts/backpropagation-srcs/Screen_Shot 19.png]]

![[AI-Notes/Concepts/backpropagation-srcs/Screen_Shot 20.png]]

Matrix multiplication is **associative** so we can compute products in any order. We do this right-to-left to avoid matrix-matrix products. We only need matrix-vector products which are more efficient.

Called Reverse Mode Auotomatic Differentiation (fancy name).

## Forward Mode Automatic Differentiation

This is done when we have a **scalar input and a vector output**.

![[AI-Notes/Concepts/backpropagation-srcs/Screen_Shot 21.png]]

![[AI-Notes/Concepts/backpropagation-srcs/Screen_Shot 22.png]]

We perform the computation left-to-right to again have only matrix-vector products.

This could be useful for a non-ML problem, such as a simulation where you can tweak the gravity of the system and want to see all the things it affects.

This isn't implemented in PyTorch or TensorFlow.

## Backprop: Higher-Order Derivatives

We can use the same back-prop algorithm to compute not just gradients, but higher order derivatives.

![[AI-Notes/Concepts/backpropagation-srcs/Screen_Shot 23.png]]

![[AI-Notes/Concepts/backpropagation-srcs/Screen_Shot 24.png]]

We can compute the second derivative of the loss with respect to the input $x_0$. This is given by the **Hessian matrix**, which tells us how fast the gradient of $x_0$ would change if we change one element of $x_1$ a little bit.

**Jacobian Matrix:**
- First derivative
- Vector in → vector out

**Hessian Matrix:**
- Second derivative
- Vector in → scalar out