---
tags: [flashcards]
source: https://www.youtube.com/watch?v=ODSJsviD_SU
summary:
---

- Musk on the affect of bots on the world: An economy is productive entities * their productivity. If there is no limit of the number of entities, what does an economy become? A future of abundance. There is no poverty (you can have whatever you want).
- It's important that the company taking the lead on changing the world is a company the public has a say in (ex. Tesla being a public company).

Some stats from Tesla:
- 35 Releases
- 281 Models Shipped
- 18,659 Pull Requests
- 75,778 Models Trained (model trained every 8 minutes)
- 4.8M Clip Dataset

![[tesla-fsd-stack.png]]
Tesla produces a vector space output for the planning stack. They have their own AI Compiler that runs on its custom chips.

### Occupancy Networks
Phil talked about Neural Networks in Tesla AI Day (1:12:46).

![[screenshot-2023-01-10_10-49-35.png]]
They predict a 3D occupancy grid with class labels. The resolution can change. The network runs every 10 ms. Occupancy flow is whether that location is predicted as moving or not.

![[screenshot-2023-01-10_10-53-36.png]]
[Source](https://youtu.be/ODSJsviD_SU?t=4500)

> [!TODO] Look into NeRF
> He thinks that NeRFs are going to be the foundation models for computer vision because you can one-shot predict a 3D volumetric construction.

### Training Infra
Tesla does a lot of stuff to improve training speed.

### Lanes
> [!TODO] What are RegNet?

They have a "special language" for encoding lane interactions. They can use language models to help solve this.

### Objects
- Predict short time horizon trajectories for all agents. Can use this to avoid collisions.

![[screenshot-2023-01-10_11-11-11.png]]

They first look at where agents are located and then based on this take additional information for those locations. Sparsification step allows for faster inference.

### Optimizing FSD
![[screenshot-2023-01-10_11-13-24.png]]
(1:38:00).