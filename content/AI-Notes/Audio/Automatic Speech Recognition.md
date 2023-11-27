---
tags: [flashcards]
source:
summary:
---
### Challenge of ASR
The goal of ASR is to detect phones which is any distinct speech sound. The same phone can sound different in different words/contexts so this makes the problem more challenging.

> [!NOTE] Phone vs Phoneme
> In phonetics, (a branch of linguistics) a phone is any distinct speech sound or gesture, regardless of whether the exact sound is critical to the meanings of words. In contrast, a phoneme is a speech sound in a given language that, if swapped with another phoneme, could change one word to another.

### Traditional ML approach vs. DL approach
The traditional approach involves manually computing features from the [[Audio Overview#Spectogram]] and then feeding those models to a traditional ML model (ex. SVC).

During training you would extract features from the speech, feed these to an MLP, and then try to predict the phone labels.
![[screenshot 2023-11-18_09_55_42@2x.png]]

During inference you would have a similar process but then use a [[Hidden Markov Model]] to predict the most likely sequence of phones (the HMM is how you make use of context).
![[screenshot 2023-11-18_09_55_47@2x.png]]

The deep learning approach just uses the spectogram directly.
![[screenshot 2023-11-18_10_01_34@2x.png]]
Left off around here https://youtu.be/D-MH6YjuIlE?t=1537.

### Current SoTA
- Wav2Vec 2.0 - Convolutional transformer + masked audio modeling.
- Conformer - Convolutional augmented transformers (models both local and global dependencies)
- ContextNet - CNN-RNN transducer network (introduces a squeeze-and-excitation layer)
