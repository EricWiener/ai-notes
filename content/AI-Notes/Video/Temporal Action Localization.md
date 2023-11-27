# Temporal Action Localization

Tags: EECS 498

Temporal action localization is when you are given a very long video sequence and you need to identify what parts of the clip belong to certain actions. You can have multiple actions throughout the clip.
![[AI-Notes/Video/temporal-action-localization-srcs/Screen_Shot.png]]

You can use an architecture similar to Faster-RCNN. You first **generate temporal proposals** (instead of region proposals) and then you **classify the proposals**.