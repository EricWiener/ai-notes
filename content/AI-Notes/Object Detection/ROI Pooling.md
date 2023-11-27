Region of interest pooling allows us to crop from a feature map in a differentiable way (so that we can backprop through it). Our bounding boxes on the original image can be different sizes, but we always want to input to our fully  connected layers (post RoI Pooling) to have the same sized inputs.

First we need to project the region proposals onto the features account for subsampling.

![[AI-Notes/Object Detection/roi-pooling-srcs/Screen_Shot 6.png]]

Then we need to align this resized area to the actual feature map (snap to the grid cells)

![[AI-Notes/Object Detection/roi-pooling-srcs/Screen_Shot 7.png]]

We then need to divide the projected region into a 2x2 grid of roughly equal subregions (in reality this is 7x7).

![[AI-Notes/Object Detection/roi-pooling-srcs/Screen_Shot_1.png]]

We then need to perform max-pooling within each region of our grid, so our output has the correct dimensions as expected by the FC layer. This can be done by taking the maximum value, bilinear sampling, or another approach.

![[AI-Notes/Object Detection/roi-pooling-srcs/Screen_Shot_2.png]]

We can now backpropagate through this in a similar way that we backpropagated through max-pooling.

**Problem:** there is a is slight misalignment of the proposed regions and the actual features they map to due to snapping to grid cells and having different sized subregions. [[RoI Align]] uses **bilinear interpolation** to get better alignment.