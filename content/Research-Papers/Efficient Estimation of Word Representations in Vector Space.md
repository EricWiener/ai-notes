---
tags: [flashcards]
source: https://arxiv.org/abs/1301.3781
aliases: [word2vec]
summary:
---

[Medium Article](https://towardsdatascience.com/word2vec-explained-49c52b4ccb71)

Word2Vec can detect synonyms because of how it represents words in a high-dimensional vector space. In the Word2Vec model, words that appear in similar contexts tend to be located close to each other in this vector space. This is based on the distributional hypothesis in linguistics, which posits that words with similar meanings often appear in similar contexts. 

When trained on a large corpus of text, Word2Vec can learn the semantic and syntactic relationships between words. As a result, synonyms, which are words with similar meanings, tend to have similar vector representations. Thus, by measuring the cosine similarity between the vectors of two words, Word2Vec can identify whether they are synonyms. 

However, it's important to note that Word2Vec isn't perfect. For example, it might mistakenly identify antonyms as similar because they also often appear in similar contexts. So while it's a powerful tool for natural language processing tasks, it also has its limitations.


### Why can you perform vector math with word2vec embeddings but not other types of embeddings?
In the case of word2vec, the model is trained to predict a word given its context or predict the context given a word (Skip-gram and CBOW respectively). During this process, the model learns to capture semantic and syntactic relationships between words, which ends up being encoded as spatial relationships in the high-dimensional space the words are embedded in (similar words will have similar embeddings since they will be used in similar contexts). This is why vector arithmetic can uncover semantic relationships, like the famous example "king" - "man" + "woman" = "queen".

Other types of embeddings like GloVe, which is trained on co-occurrence statistics in a corpus, also support vector operations as they capture similar kinds of relationships in their training process.

However, not all types of embeddings support meaningful vector operations. For example, one-hot encoded word vectors, which simply represent each word as a binary vector with a 1 in the position corresponding to that word's index, do not support meaningful vector arithmetic. This is because they don't capture any semantic or syntactic relationships between words during their generation process.

Even with more advanced embedding methods like BERT (Bidirectional Encoder Representations from Transformers), vector arithmetic might not hold the same intuitive relationships, because these models aim to embed words in context, leading to dynamic embeddings. In other words, the same word can have different embeddings depending on its context.

It is also important to remember that even for models like word2vec, while many vector operations do seem to uncover meaningful relationships, this doesn't hold true universally. The operations are reliant on the relationships learned during training, and if certain relationships aren't well-represented in the training data, the model won't be able to capture them.

# Paper Notes
The paper proposed two novel model architectures (Continuous Bag-of-Words and Continuous Skip-gram Model) for computing continuous vector representations of words from very large datasets.

![[word2vec-model-architectures.png]]
>New model architectures. The CBOW architecture predicts the current word based on the context, and the Skip-gram predicts surrounding words given the current word.

The goal of the paper was to learn high-quality word vectors from a huge dataset. They measured their performance using a word similarity task and trying to maximize the accuracy of vector operations with word embeddings (ex. ` vector(”King”) - vector(”Man”) + vec tor(”Woman”) = vector("Queen")`) while minimizing the computational complexity with regards to epochs, number of words in the training set, and model complexity.

### Previous Work
Some previous work treats words as atomic units (one hot encoding in a vector of length $V$ where $V$ is the number of words in the vocabulary). This is useful for simplicity, but the simple techniques had reached their limits in many tasks and "there are situations where simple scaling up of the basic techniques will not result in any significant progress, and we have to focus on more advanced techniques."

> [!NOTE] This reminds me a lot of current approach to training larger and larger language models
> there are situations where simple scaling up of the basic techniques will not result in any significant progress, and we have to focus on more advanced techniques.


### Model Training
They train the model in less than a day with a 1.6 billion word dataset. They use distributed CPU training to train on 100 CPU cores.

### Continuous Bag-of-Words Model
This model tries to predict the current word based on the context from the four future and four historical words.

### Continuous Skip-gram Model
This model tries to predict words in a certain range before and after the current word. They found that increasing the range improves quality of the resulting word vectors, but it also increases the computational complexity.