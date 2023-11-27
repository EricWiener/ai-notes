---
tags:
  - flashcards
source: 
aliases:
  - NMS
summary: used to decide which bounding boxes to keep as the final output of an object detector by keeping high-confidence non-overlapping boxes.
---
We will get many bounding boxes as the output of our detection. Some of these bounding boxes might be very similar and will have overlap. We need to decide which to keep.

We will only keep "peaks" in the detector response and discard low probability bounding boxes that are near high probability ones.

You can do this with a greedy algorithm:

![[AI-Notes/Concepts/object-detection-srcs/Screen_Shot 13.png]]

**Algorithm:**
1. Select next highest-scoring box
2. Add it to your set of boxes to keep if it doesn't significantly overlap any of your already selected boxes. Otherwise, discard.
3. If any boxes remain, repeat from 1.

![[example-image-where-nms-would-do-poorly.png|300]]

**Problem:** NMS may eliminate "good" boxes when objects are highly overlapping (like in the above image). There is no good solution to this problem yet.
