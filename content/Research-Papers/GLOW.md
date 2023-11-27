---
tags:
  - flashcards
source: https://openai.com/research/glow
summary: "[[Normalizing Flows]] model from OpenAI that generates some very cool images of faces"
---
This paper is a  reversible generative model ([[Normalizing Flows]]) that introduces a reversible 1x1 convolution and allows you to generate realistic and high quality images. You can also interpolate between images and manipulate images via manipulation in the latent space.
### Interpolation in latent space
You can encode two images to get two vectors in the output distribution. You can then sample from intermediate points between these two embeddings to do things like interpolate between arbitrary faces. See [here](https://cdn.openai.com/research-covers/glow/videos/both_loop_new.mp4) for an example.

> [!PRIVTE] Visualization Video
> ![[prafulla_people_loop.mp4|300]]

### Manipulation in latent space
You can manipulate images by manipulating the latent space. These semantic attributes could be the color of hair in a face, the style of an image, the pitch of a musical sound, or the emotion of a text sentence. Since flow-based models have a perfect encoder, you can encode inputs and compute the average latent vector of inputs with and without the attribute. The vector direction between the two can then be used to manipulate an arbitrary input towards that attribute.

The above process requires a relatively small amount of labeled data, and can be done after the model has been trained (no labels are needed while training). The following code snippet shows how this can be done:
```python
# Train flow model on large, unlabelled dataset X
m = train (X_unlabelled)

# Split labelled dataset based on attribute, say blonde hair
X_positive, X_negative = split(X_labelled)

# Obtain average encodings of positive and negative inputs
z_positive = average([m.encode(x) for x in X_positive])
z_negative = average([m.encode(x) for x in X_negative])

# Get manipulation vector by taking difference
z_manipulate = z_positive - z_negative

# Manipulate new x_input along z_manipulate, by a scalar alpha \in [-1, 1]
z_input = m.encode(x_input)
x_manipulated = m.decode(z_input + alpha * z_manipulate)
```