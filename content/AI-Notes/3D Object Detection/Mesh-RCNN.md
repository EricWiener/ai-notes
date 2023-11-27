# Mesh-RCNN

Tags: EECS 498, Model

Mesh R-CNN is a model that Professor Johnson created. 

**Input:** a single RGB image

**Output:** a set of detected objects. For each object, it gives a:

- Bounding box
- Category label
- Instance segmentation
- 3D triangle mesh

The bounding box, category label, and instance segmentation are all predicted using Mask R-CNN. They then add a mesh head that can predict the 3D triangle mesh.

![[Mesh refinement]]

Mesh refinement

Most models that predict triangle meshes use mesh deformation. They start with an initial ellipsoid (or similar shaped) mesh and then refine it. However, this approach doesn't work well when their are holes in the object because you can't create holes by refining the mesh.

To resolve this, they first predict 3D object voxels based on the mesh and then refine those voxel predictions to create the final 3D object mesh.

![[AI Notes/3D Object Detection/Mesh-RCNN/Screen_Shot 1.png]]

This allows the model to handle objects with holes in them much better.

![[AI Notes/3D Object Detection/Mesh-RCNN/Screen_Shot 2.png]]