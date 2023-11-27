# Canonical vs. View Coordinates

**Canonical coordinates** predict the 3D shape in a canonical coordinate system (e.g. the front of the chair is always +z) regardless of the viewpoint of the input image. You need to decide before hand what each axis of the object will be. 

**View coordinates:** predict 3D shapes aligned to the viewpoint of the camera (what the image looks like).

![[input-canonical-target.png.png]]

Here you can see that given an input image, the canonical target will always output a 3D shape in the same position. The view target, on the other hand, will match the orientation of the input image.

Most papers predict in canonical coordinates because it makes it easier to load data.  However, the canonical view breaks the **"principle of feature alignment" that says predictions should be aligned to inputs.**

View coordinates maintain alignment between inputs and predictions. In general, **you should prefer view coordinate systems** because identical models trained with both systems tend to perform better with view coordinates.

The canonical view overfits to training shapes, so it has a better generalization to new views of known shapes, but worse generalization to new shapes or new categories.