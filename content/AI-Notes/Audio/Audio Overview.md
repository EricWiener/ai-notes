---
tags:
  - flashcards
source: https://youtu.be/D-MH6YjuIlE
summary: Overview of how audio data works and can be represented
---
[Slides](https://github.com/Vaibhavs10/ml-with-audio/blob/master/slides/ml-4-audio-session2.pdf)
### What is sound?
When someone speaks, molecules in the air gets displaced. The change in the air pressure is picked up by your ears and is interpreted as sound.
![[intro-to-audio-and-asr-deep-dive-20231117112909265.png]]
The above example shows one sinusoidal wave, but in reality sound is a lot more complicated and looks like the following which is called a **waveform**:
![[screenshot 2023-11-17_11_31_28@2x.png|500]]
### Analog to digital conversion
You need to sample and quantize the analog sound to convert it into a digital format.
- Sampling: sample at regular points in time. The higher your sampling rate the closer you will be to the original sound.
- Quantization: amplitude is represented in bits and based on your precision you will get a better/worse approximation of the original amplitude.

![[screenshot 2023-11-17_11_35_12@2x.png|400]]
### Intensity and Loudness
- Intensity is the rate at which energy is transferred. It is measured in decibels and a 10x increase in the energy of wave results in a 10 dB increase of intensity.
- Loudness is the subjective perception of sound and depends on many factors (ex. age, background noise, distance).
### stft
You can decompose a complex input signal into a set of waves using a short-time Fourier Transform. A Fourier Transform works by representing a function as a sum of sinusoidal functions with different frequencies.

If we transform a spoken sentence to the frequency domain, we obtain a spectrum which is an average of all phonemes in the sentence, whereas often we would like to see the spectrum of each individual phoneme separately ([source](https://wiki.aalto.fi/display/ITSP/Spectrogram+and+the+STFT)). Therefore, you use the short-time Fourier Transform to transform small windows of the signal.
### Spectogram
After performing the transform, you can visualize the results with a **spectogram** which shows the frequency (Hz) at each time window (column). The intensity is also shown via the color (dB).

![[screenshot 2023-11-18_09_29_17@2x.png|300]]  ![[screenshot 2023-11-18_09_32_49@2x.png|300]]
### Mel Spectogram
Humans do not perceive changes in frequency linearly (going from 100->200 Hz conveys as much info as going from 10k->20k Hz).

The unit **Mel** is used where equal distances in pitch sound equally distant to the listener. You can map your frequency values to the Mel scale and then plot these values. This is called a ==Mel Spectogram==.
![[screenshot 2023-11-18_09_38_40@2x.png|400]]
<!--SR:!2023-12-15,15,290-->

