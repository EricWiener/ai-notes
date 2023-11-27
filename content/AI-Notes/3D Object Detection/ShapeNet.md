# ShapeNet

![[shapenet.png.png]]

ShapeNet comprises of ~50 categories and ~50k 3D CAD models. This is supposed to be the ImageNet of 3D shapes. Each of the models are rendered from multiple viewpoints to create lots of training images (~1 million).

Many papers use it to show results, but it is very synthetic. It has lots of isolated objects with no contents. Additionally, it has lots of chairs, cars, and airplanes. Not a lot of diversity.

# Pix3D

![[AI-Notes/3D Object Detection/datasets-imgs/Screen_Shot 1.png]]

This consists of 9 categories with 219 3D models of IKEA furniture. It is aligned to ~17K real images. This dataset was collected by researchers at MIT. People like to post pictures of their IKEA furniture online and IKEA posts the 3D meshes for their furniture. They looked up IKEA furniture pictures and then paid people to align the meshes to the images.

It's nice because it has real images and context, but the dataset is also small and has partial annotations (only 1 object per image).