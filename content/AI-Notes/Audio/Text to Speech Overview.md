---
tags:
  - flashcards
source: https://youtu.be/aLBedWj-5CQ
summary:
---
These are my notes from the [Hugging Face ML for Audio Study Group](https://youtu.be/aLBedWj-5CQ). They make reference to the textbook [SLP3](https://web.stanford.edu/~jurafsky/slp3/) chapter 16.

Some implementations:
- https://github.com/coqui-ai/tts

# Why is TTS hard?
**The same word can have different pronounciations in different contexts:**
- "It's no **use** to ask to **use** the telephone"
- "Do you **live** near a zoo with **live** animals?"
- "I prefer **bass** fishing to playing the **bass** guitar"

This also extends to things like whether you are asking a question or making a statement.

**You need to handle non-standard words:**
1. Numbers
2. Monetary amounts
3. Abbreviations
4. Dates (dates need to be read out - you can't just read the digits and slashes).
5. Acronyms

**seventeen fifty**: (in The European economy in 1750")
**one seven five zero**: (in "The password is 1750")
**seventeen hundred and fifty**: (in "1750 dollars")
**one thousand, seven hundred, and fifty**: (in "1750 dollars")

**How is this handled?**
1. You can use regular expressions to match things like dates or monetary amounts.
2. You can use a [[RNN, LSTM, Captioning#Sequence to Sequence (seq2seq)]] model to convert a sequence into another sequence that the audio predicting model takes in.

# [[Audio Overview#Mel Spectogram|Mel Spectogram]] Prediction
This has a similar architecture as is used for ASR (automatic speech recognition). You have an encoder-decoder with attention.

The encoder takes a sequence of letters and produces a hidden representation representing the letter sequence.

The hidden representation is then used by the attention mechanism in the decoder.

https://youtu.be/aLBedWj-5CQ?t=906