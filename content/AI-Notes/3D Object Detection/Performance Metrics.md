# Performance Metrics

Tags: EECS 498

# Voxel IoU

In 2D, we used IoU as a metric to compare models. However, generalizing this to 3D doesn't work very well. 

- You cannot capture thin structures
- Cannot be applied to pointclouds
- For meshes, need to voxelize or sample
- Not very meaningful at low values
    
    ![[Diagram showing that low 3D IoU scores don't tell you much (why does a lead blower have a higher score than a table)]]
    
    Diagram showing that low 3D IoU scores don't tell you much (why does a lead blower have a higher score than a table)
    

# Chamfer Distance

![[AI Notes/3D Object Detection/Performanc/Screen_Shot 1.png]]

You could convert your prediction and ground-truth into pointclouds via sampling. These pointclouds could then be compared with Chamfer distance. 

However, because Chamfer distance uses the squared distance, it is very sensitive to outliers.

The upside to using Chamfer distance is it can be directly optimized.

# F1 Score

This is similar to the Chamfer loss because you convert your shapes to pointclouds via sampling.

Here, the orange points are the predicted ones and the blue are the ground-truth.

![[AI Notes/3D Object Detection/Performanc/Screen_Shot 2.png]]

You can then calculate the **precision** by looking at how many of the predicted points fall close to a ground-truth point. You can draw a sphere around the predicted points and if the ground-truth point falls inside the sphere, it will be considered correct.

Here the precision is 3/4 because 3 out of the 4 predicted points have a ground-truth point within their sphere.

![[AI Notes/3D Object Detection/Performanc/Screen_Shot 3.png]]

You can then look at the **recall**, which is the fraction of ground-truth points that have a predicted point within their sphere. 

Here the recall is 2/3 because 2 out of the 3 ground-truth points have an orange point within their sphere (the bottom predicted point doesn't actually fall within the ground-truth's sphere).

![[AI Notes/3D Object Detection/Performanc/Screen_Shot 4.png]]

![[AI Notes/3D Object Detection/Performanc/Screen_Shot 5.png]]

You can then compute the F1 score as a weighted combination of the precision and recall scores. The only way to get an F1 score of 1.0 is if the point clouds are overlapping entirely. It is more robust to outliers (no squaring) and is probably the best shape prediction metric in common use.

The one downside is you need to look at different threshold values (sphere sizes) to capture details at different scales.