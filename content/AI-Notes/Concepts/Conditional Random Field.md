---
tags: [flashcards]
aliases: [CRF]
source: https://towardsdatascience.com/conditional-random-fields-explained-e5b8256da776
summary: Conditional Random Fields is a class of discriminative models best suited to prediction tasks where contextual information or state of the neighbors affect the current prediction.
---

A conditional random field (CRF) is a statistical model that is used for structured prediction. Structured prediction is the task of predicting a sequence of labels, such as the parts of speech in a sentence or the tags in a document. CRFs are a type of discriminative model, which means that they are trained to directly predict the labels, rather than to generate a probability distribution over the labels.

CRFs are built on top of a graphical model, which is a mathematical representation of the dependencies between the labels. The graphical model for a CRF is a directed acyclic graph (DAG), where each node represents a label and each edge represents a dependency between two labels.

The parameters of a CRF are learned using a maximum-likelihood estimator. The maximum-likelihood estimator finds the parameters that maximize the probability of the observed labels.
