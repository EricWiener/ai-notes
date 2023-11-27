---
tags: [flashcards]
source: https://www.msri.org/people/members/eranb/Combining_td_and_bu.pdf
summary:
---

The top-down approach uses object representation learned from examples to detect an object in a given input image and provide an approximation to its figure-ground segmentation. The bottom- up approach uses image-based criteria to define coherent groups of pixels that are likely to belong together to either the figure or the background part.

![[top-down-vs-bottom-up-segmentation.png]]
> (a.) Input image. (b.) The bottom-up hierarchical segmentation (at three different scales) can accurately detect image discontinuities but may also segment an object into multiple regions and merge object parts with the background. (Each region in uniform color represents a segment.) (c.) The top-down approach provides a meaningful approximation for the figure- ground segmentation of the image, but may not follow exactly image discontinuities.

### Bottom-up
The bottom-up approach, is to first segment the image into regions and then identify the image regions that correspond to a single object. Relying mainly on continuity principles, this approach groups pixels according to the grey level or texture uniformity of image regions, as well as the smoothness and continuity of bound- ing contours. The main difficulty of this approach is that an object may be segmented into multiple regions, some of which may merge the object with its background.

### Top-down
Top-down segmentation, is therefore to use prior knowledge about an object, such as its possible shape, color, or texture, to guide the segmentation. The main difficulty in this approach stems from the large variability in the shape and appearance of objects within a given class