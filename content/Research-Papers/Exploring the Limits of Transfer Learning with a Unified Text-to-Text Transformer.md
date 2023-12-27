---
tags:
  - flashcards
source: https://arxiv.org/abs/1910.10683
summary: Google did a survey of various techniques to improve transformer performance and then used the results of the survey to create a SOTA model that generalizes to many types of tasks.
aliases:
  - T5
publish: true
---
![[Research-Papers/exploring-the-limits-of-transfer-learning-with-a-unified-text-to-text-transformer-srcs/screenshot 2023-12-04_11_36_05@2x.png]]
The paper aims to explore the existing set of transfer learning techniques for NLP by treating all tasks (ex. translation, question answering, classification, etc.) as feeding the model text as input and training it to generate some target text. This allows using the same model, loss function, hyperparameters, etc. across a diverse set of tasks so they can compare the effectiveness of different factors. "T5" refers to the model which they call "**T**ext-**t**o-**T**ext **T**ransfer **T**ransformer" (5 T's there).

# Introduction
- Computer vision pretraining is often done via supervised learning on a large labeled dataset. NLP typically uses unsupervised learning on unlabeled data.
- Unsupervised pre-training for NLP is attractive because unlabeled text data is available in large quantities on the internet.
- The paper doesn't aim to propose new methods and instead aims to conduct a comprehensive survey on where the field stands.
- Based on the results of their survey, they are able to create a model that achieves SOTA on many tasks.
- They release the "Colossal Clean Crawled Corpus" (100s of GB of clean English text scraped from the web), code, and pre-trained models. Also referred to as C4. [HuggingFace Link](https://huggingface.co/datasets/c4)

# Setup
- Early NLP transformer learning used RNNs but now it is popular to use [[AI-Notes/Transformers/Transformer|Transformer]]s.
- The model is similar to the original transformer but removes the [[AI-Notes/Layers/Layernorm|Layernorm]] bias, places the layer normalization outside the residual path, and uses a different position embedding scheme.
- They train the models on TPU supercomputers.
### Training Dataset
[[Research-Papers/Datasets/CommonCrawl|CommonCrawl]] contains a lot of un-useful text like menus, error messages, or duplicate text. They cleaned up web extracted text from April 2019 and released a new 750 GB dataset called the **C**olossal **C**lean **C**rawled **C**orpus.

Some interesting things they did were:
- Many of the scraped pages contained warnings stating that Javascript should be enabled so we removed any line with the word Javascript.
- Many pages had boilerplate policy notices, so we removed any lines containing the strings “terms of use”, “privacy policy”, “cookie policy”, “uses cookies”, “use of cookies”, or “use cookies”.
### Downstream Tasks
They measure performance on multiple benchmarks including machine translation, question answering, summarization, and text classification.

They measure performance on the [[Research-Papers/Datasets/GLUE and SuperGLUE|GLUE and SuperGLUE]] text classification meta-benchmarks; CNN/Daily Mail abstractive summarization; SQuAD question answering; and WMT English to German, French, and Romanian translation.

### Input and Output Format
When pre-training and fine-tuning they use a "text-to-text" format where they add a task specific prefix to the original input sentence before feeding it to the model. This allows using a consistent training objective for pre-training and fine-tuning where the model is trained to maximize the likelihood of the text regardless of the task.

The exact wording of the prefix seemed to have little impact so they just fixed one per task as a hyperparameter.

See Appendix D for examples from the paper.

**Translation:**
> To ask the model to translate the sentence “That is good.” from English to German, the model would be fed the sequence “translate English to German: That is good.” and would be trained to output “Das ist gut.”

**Text Classification**
> For text classification tasks, the model simply predicts a single word corresponding to the target label. For example, on the MNLI benchmark the goal is to predict whether a premise implies (“entailment”), contradicts (“contradiction”), or neither (“neutral”) a hypothesis. With our preprocessing, the input sequence becomes “mnli premise: I hate pigeons. hypothesis: My feelings towards pigeons are filled with animosity.” with the corresponding target word “entailment”.

More examples of text classification:
- **Entailment** (premise implies hypothesis): John ate pasta for supper $\implies$ John ate pasta for dinner.
- **Contradiction** (premise contradicts the hypothesis): The market is about to get harder, but possible to navigate.	 $\not \implies$ The market is about to get harder, but not impossible to navigate.	
- **Neutral**: All dogs like to scratch their ears ~ All animals like to scratch their ears.

# Experiments
They evaluate a variety of different techniques (ex. pre-training objectives, model architectures, datasets, etc.). Since there are so many possible combinations, they only alter one technique at a time (ex. assess different unsupervised objectives while keeping the rest of the pipeline fixed). This reduces the space they need to search but also means they may miss ideal combinations (ex. using a certain dataset might work better with a larger model).

### Modifying models
They modified some existing approaches in order to get the models to work with the text-to-text format.

For instance, [[Research-Papers/BERT|BERT]] is an encoder-only model and either predicts the most likely next token or predicts the most likely next sequence. This is sufficient for text classification (predict a single token label) or [[AI-Notes/Natural Language Processing/Span|Span]] prediction tasks (ex. predict the full subject in "*The Cleveland Indian football team* lost the game") but it does't work for translation of summarization.

Therefore, they consider a model that behaves similar to [[Research-Papers/BERT|BERT]] but not exactly the same (it is trained with a similar [[Research-Papers/BERT#Masked LM (Masked Language Modeling)]] objective.

> [!NOTE] There is a large portion of the paper that I did not include notes on.

### Architecture
An encoder-decoder architecture with a denoising objective (you mask certain tokens and then train the model to predict what the masked tokens are) performs best. They shared parameters across the encoder and decoder. Reducing the number of layers in the encoder-decoder hurt performance.

### Objective
They found most denoising objectives which train the model to reconstruct randomly corrupted text, performed similarly in the text-to-text setup. As a result, they suggest using objectives that produce short target sequences so that unsupervised pre-training is more computationally efficient.

**3.3.4: Corrupting Spans:**:
> We now turn towards the goal of speeding up training by predicting shorter targets. The approach we have used so far makes an i.i.d. decision for each input token as to whether to corrupt it or not. When multiple consecutive tokens have been corrupted, they are treated as a “span” and a single unique mask token is used to replace the entire span. Replacing entire spans with a single token results in unlabeled text data being processed into shorter sequences. Since we are using an i.i.d. corruption strategy, it is not always the case that a significant number of corrupted tokens appear consecutively. As a result, we might obtain additional speedup by specifically corrupting spans of tokens rather than corrupting individual tokens in an i.i.d. manner.

Ex: if your full sentence is "I received a heavy fine but it failed to crush my spirit." then you could randomly mask tokens and end up with: "I received a **heavy fine** but it **failed** to crush my spirit." where the bold is masked. This becomes "I received a {x} but it {y} to crush my spirit." where the model now needs to predict {x} and {y}. If instead you did "I received a **heavy fine** **but** it failed to crush my spirit." then you only need to predict a single token for {x} where {x} is "heavy fine but."
# Takeaways
- In spite of its simplicity, we found the text-to- text framework obtained comparable performance to task-specific architectures and ultimately produced state-of-the-art results when combined with scale.
- An encoder-decoder model works best (vs. encoder-only and decoder-only). You can share parameters between the encoder and decoder without a significant performance drop.
- Using a larger and diverse dataset improves performance.
- Updating all model parameters during fine-tuning outperformed methods to update fewer parameters (ex. [[Research-Papers/Parameter-Efficient Transfer Learning for NLP|Adapter Modules]]).
- For the denoising objective if you corrupt a sequence of tokens and then combine consecutive masked tokens into a single token to be predicted, then you can speed up training by purposefully masking sequences to predict instead of masking tokens randomly. This results in fewer tokens you need to predict which speed up training.

