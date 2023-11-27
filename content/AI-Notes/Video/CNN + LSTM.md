# CNN + LSTM

Tags: EECS 498, Model

Using a traditional 3D CNN, your filters are only applied over a small amount of time for each convolution. We might want to be able to capture longer trends over time. We can do this by first extracting features from the image with 2D CNNs (or 3D CNNs over small time chunks) and then using an LSTM (or other recurrent network) to fuse the local features temporally. 

You can use a many-to-one model to make a prediction for the entire video using the last hidden state of the LSTM. You could also use a many-to-many model to make predictions over the sequence of frames.

![[AI-Notes/Video/cnn+lstm-srcs/Screen_Shot.png]]

**Back-propagation:**

One trick to save memory is to not backprop into the CNN. You can pre-train it and use it as a feature extractor.