---
tags:
  - flashcards
source: 
summary:
---
[Great YouTube Video](https://youtu.be/Cs6pe8o7XY8)

Gemini 1.5 Pro is much better at writing (based on a human analysis) than GPT-4 and shows up as less likely to be AI generated using tools like [binoculars](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbEN5dV9sbzZxdUhULXBaTUhHN3dESTZfeHEwUXxBQ3Jtc0tudXNtbnI1SDdseUhWTkhyZlh3emFRWVhIR2hZY21qLTA2aHNoZEJpVm1VWGZKcThqdlM2U01RZTBrYTNzQWlDWTJINGN5amxsWFI5RS1GTGxfZXVGWHJhSTdGZTYtYjBHQTNZdFF4SXRMYVV6cExEaw&q=https%3A%2F%2Fhuggingface.co%2Fspaces%2Ftomg-group-umd%2FBinoculars&v=Cs6pe8o7XY8).
# [Google Blog Post](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/#gemini-15)
- The models can run up to 1 million tokens consistently, achieving the longest context window of any large-scale foundation model yet.
- Gemini 1.5 is more efficient to train and serve than Gemini 1.0, with a new [Mixture-of-Experts](https://arxiv.org/abs/1701.06538) (MoE) architecture.
- Gemini 1.5 Pro comes with a standard 128,000 token context window and they are working on rolling out the full 1 million token context window
- Gemini 1.5 is built upon our leading research on [Transformer](https://blog.research.google/2017/08/transformer-novel-neural-network.html) and [MoE](https://arxiv.org/abs/1701.06538) architecture. While a traditional Transformer functions as one large neural network, MoE models are divided into smaller "expert” neural networks.
- MoE models learn to selectively activate only the most relevant expert pathways in its neural network. This specialization massively enhances the model’s efficiency.
- Our latest innovations in model architecture allow Gemini 1.5 to learn complex tasks more quickly and maintain quality, while being more efficient to train and serve.
- In the [Needle In A Haystack](https://github.com/gkamradt/LLMTest%5FNeedleInAHaystack) (NIAH) evaluation, where a small piece of text containing a particular fact or statement is purposely placed within a long block of text.
# [Technical Report](https://goo.gle/GeminiV1-5)
- Gemini 1.5 Pro can achieve near-perfect retrieval on up to at least 10M tokens.
- Gemini 1.5 Pro matches or surpasses Gemini 1.0 Ultra's performance on a broad set of benchmarks.
- The model uses a new mixture of experts architecture.
- Gemini 1.5 Pro achieves almost perfect performance on the [Needle In A Haystack](https://github.com/gkamradt/LLMTest%5FNeedleInAHaystack) (NIAH) evaluation up to 1M tokens. It outperforms other models (ex. GPT-4) even when these models are augmented with external retrieval methods (RAG).
- In non-long context tasks, Gemini 1.5 Pro doesn't blow away the competition as much.
### Comparison to Gemini 1.0 Pro and Ultra
![[Research-Papers/assets/Gemini 1.5/screenshot_2024-02-16_11_58_51@2x.png|400]]
- Win-rate is how many benchmarks the model performed better on / the total number of benchmarks. Ex: win-rate of 77% is just 10/13 * 100 = 77.
- Gemini 1.5 Pro greatly improves over 1.0 Pro. It performs on-par and worse than 1.0 Ultra for some areas. However, it is trained with less compute and is more efficient at inference.

### Model Architecture
- Gemini 1.5 Pro is a sparse mixture-of-experts Transformer based model that builds on Gemini 1.0.
- AI Insider suspects that improvements from [[Research-Papers/Mixtral of Experts|Mixtral of Experts]] were used to improve long context performance.
- The model was trained across multiple data centers (each data center has around 32k GPUs).

### Results:
**Kalamang:**
They gave Gemini 1.5 Pro a reference grammar book for a language with very few speakers. No Kalamang was in the training data. Gemini 1.5 Pro was able to perform on the same level as someone who had learned from the grammar book.

**Understanding Long Documents:**
![[Research-Papers/assets/Gemini 1.5/screenshot_2024-02-16_12_18_03@2x.png|500]]
Gemini 1.5 Pro continues to improve understanding on extremely long documents.

**Video QA**:
They needed to make their own benchmark to evaluate the model since existing benchmarks were too easy.

**OCR**:
Gemini 1.5 Pro does not outperform 1.0 Pro on OCR.
### Impacts
- You can understand longer context. This could be used to explore archival content which could help journalists to historians.
- Could improve memory of long conversations with LLMs.
- Could understand long YouTube videos.

### Safety
- Gemini 1.5 Pro has a higher refusal rate where it refuses to answer questions for safety reasons.
### Memorization
To test for memorization, we use a similar methodology as Nasr et al. (2023), which attacks the model by asking it to repeat a single token many times. When successful, this attack first causes the model to diverge, i.e., to output text that is not a repetition of the specified token. We then examine which of these diverged outputs may contain regurgitated training data.

It is easier to obtain memorized data with longer prompts.

