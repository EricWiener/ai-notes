---
tags: [flashcards, eecs498-dl4cv]
source:
summary:
---

![[AI-Notes/Video/slowfast-srcs/Screen_Shot_2.png]]

SlowFast is the current state of the art in terms of working with videos. The two-stream networks rely on the hand-crafted features of optical flow. We want to get rid of that requirement and build a network that can just train on the raw pixel values.

You have two parallel network branches. Both branches operate on the raw pixels, but they use different temporal resolutions. 

- The slow branch operates at a very low frame rate, but it has more processing and a lot of channels.
- The fast branch operates at a high frame rate, but it has less processing and less channels.
- You then have lateral connections that fuse the information from the two channels
- Predictions are made using the outputs of both paths at the end