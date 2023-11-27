# Flattened Convolutional Neural Networks for Feedforward Acceleration

PDF: https://drive.google.com/open?id=1Ezt0JHt6UX1My_0-9bLv_w6Fgza_G6I_&authuser=ecwiener%40umich.edu&usp=drive_fs
Reviewed: Yes
summary: Seperates 3D Conv into three consecutive 1D filters: channels, vertical, and horizontal. MobileNet uses a similar approach, but only splits into three 2D kernels
url: https://arxiv.org/abs/1412.5474

### Abstract + Intro

- Designed for fast feed-forward execution
- There are a lot of redundant parameters in Conv networks. These cause unnecessary computation and degrade learning capacity.
- This paper uses a consecutive sequence of one-dimensional filters across all directions in 3D space.
- Found the flattened layer can effectively substitute for the 3D filters without loss of accuracy and provides a **two times speed-up** during the feedforward pass
- No additional tuning or post processing is needed.

### Related Work

- Due to the arbitrary locations and shapes of sparse filters (as generated using L1 weight regularization to induce edge connections of 0 and then skipping these 0 connections), in practice it is difficult to take advantage of sparsity with highly parallelized processing pipelines.
- Denton et al. (2014): Pre-trained 3D filters are approximated to low rank filters and the error is minimized by using clustering and post training to tune the accuracy. Their method demonstrates speedup of convolutional layers by a factor of two, while keeping the accuracy within 1% of the original model.
- Denil et al. (2013) demonstrates that 5% of essential parameters can predict the rest of parameters in the best case

### Results

- We found that removing one direction from the LV H pipeline caused a significant accuracy loss, which implies that convolutions toward all directions are essential.
- 1D convolutions is more vulnerable to vanishing gradient problem. Longer gradient path (since the convs are split up) experiences more steps of parameter updates and error accumulation, which possibly cause fast decaying gradients
- Two times speed-up in evaluation, ten times less parameters, and similar results.