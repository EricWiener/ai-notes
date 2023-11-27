---
tags: [flashcards]
source: https://learnopencv.com/image-matting-with-state-of-the-art-method-f-b-alpha-matting/
summary: the trimap is a rough segmentation of an image into three region types - certain foreground, unknown, certain background.
---

![[trimap-example.png]]
The above image shows trimaps of different sizes. There are three labels (black, gray, and white) corresponding to background, unknown, and object boundary.

### Trimap Generation
![[screenshot-2022-12-14_09-38-36.png]]
[This](https://github.com/lnugraha/trimap_generator_) package will generate trimaps for you based on an input image and how thick you want the band to be (how many pixels to dilate).

They do this by applying a $3 \times 3$ kernel 
```python
def scaling(self, image, dilation):    
    # Design an odd-sized erosion kernel
    kernel = np.ones((3,3), np.uint8)
    
    # The number of dilations
    image  = cv2.dilate(image, kernel, iterations=dilation)

    # Any gray-clored pixel becomes white (smoothing)
    image  = np.where(image > 0, 255, image)
```
[Source](https://github.com/lnugraha/trimap_generator/blob/a279562b0d0f387896330cf88549e67618d1eb7f/src/foreground_scaling.py#L74). You could also do this by calling `cv2.dilate` manually.