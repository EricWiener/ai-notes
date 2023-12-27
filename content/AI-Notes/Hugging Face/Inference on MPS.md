---
tags: [flashcards]
source:
summary:
---
I timed how long it takes a batch to run on MPS for different batch sizes. `animals` was a list of 30 animals. It's interesting how the inference speed is the fastest after batch size 30. Also interesting how there was a big speed up around batch size = 5 but then goes back up.
```python
import time
import matplotlib.pyplot as plt

inputs = [f"Tell me a joke about {animal}." for animal in animals]

def time_batch(batch_size: int) -> float:
    start_time = time.time()
    for out in pipe(inputs, batch_size=batch_size, truncation="only_first"):
        continue
    end_time = time.time()
    return end_time - start_time

batch_sizes = list(range(1, 41, 2))
times = [time_batch(bs) for bs in batch_sizes]

plt.figure(figsize=(10, 5))
plt.plot(batch_sizes, times, marker='o')
plt.xlabel('Batch Size')
plt.ylabel('Time Taken (sec)')
plt.title('Time taken for different batch sizes')
plt.grid(True)
plt.show()
```

![[AI-Notes/Hugging Face/inference-on-mps-srcs/inference-on-mps-20231224163339778.png]]

