---
tags: [flashcards]
source: https://machinelearningmastery.com/transduction-in-machine-learning/
---

**Transduction (Generally)**:
Transduction is about converting a ==signal into another form==. 
<!--SR:!2024-07-01,607,290-->

It is a popular term from the field of electronics and signal processing, where a “_transducer_” is a general name for components or modules converting sounds to energy or vise-versa.

> All signal processing begins with an input transducer. The input transducer takes the input signal and converts it to an electrical signal. In signal-processing applications, the transducer can take many forms. A common example of an input transducer is a microphone.

**Transduction in Sequence Prediction**:
A transducer is narrowly defined as a model that outputs one time step for each input time step provided. This maps to the linguistic usage, specifically with finite-state transducers.

> Another option is to treat the RNN as a transducer, producing an output for each input it reads in.

> Many natural language processing (NLP) tasks can be viewed as transduction problems, that is learning to convert one string into another. Machine translation is a prototypical example of transduction and recent results indicate that Deep RNNs have the ability to encode long source strings and produce coherent translations

The following is a list of transductive natural language processing tasks:
-   **Transliteration**, producing words in a target form given examples in a source form.
-   **Spelling Correction**, producing correct word spelling given incorrect word spelling.
-   **Inflectional Morphology**, producing new sequences given source sequences and context.
-   **Machine Translation**, producing sequences of words in a target language given examples in a source language.
-   **Speech Recognition**, producing sequences of text given sequences of audio.
-   **Protein Secondary Structure Prediction**, predicting 3D structure given input sequences of amino acids (not NLP).
-   **Text-to-Speech**, or speech synthesis, producing audio given text sequences.

Some new methods are explicitly being named as such. Navdeep Jaitly, et al. refer to their new RNN sequence-to-sequence prediction method as a “_Neural Transducer_“, which technically RNNs for sequence-to-sequence prediction would also be.

> we present a Neural Transducer, a more general class of sequence-to-sequence learning models. Neural Transducer can produce chunks of outputs (possibly of zero length) as blocks of inputs arrive – thus satisfying the condition of being “online”. The model generates outputs for each block by using a transducer RNN that implements a sequence-to-sequence model.