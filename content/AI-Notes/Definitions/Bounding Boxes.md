---
tags: [flashcards]
source:
summary:
---

![[axis-aligned-and-oriented-bbox.png]]
- Bounding boxes are typically axis-aligned (blue)
- Oriented boxes (green) are much less common since this requires extra work in the model architecture and it is harder to annotate the dataset.

**Modal vs. Amodal Boxes**
![[modal-vs-amodal-bbox.png]]
- Modal bounding boxes usually cover only the visible portion of the object (blue).
- ==Amodal detection== bounding boxes cover the entire extent of the object, even occuluded parts (green).
<!--SR:!2024-03-02,294,310-->

- The term "amodal" comes from Psychology where amodal perception is a field of study looking at how humans perceive parts of objects we can't see.
- "Modal" is just not amodal.