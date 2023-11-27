---
tags: [flashcards]
source: https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215
summary: filters are seperated into different groups. This allows splitting weights to different GPUs (AlexNet), reduces parameters, and increases cardinality (ResNext).
---

In grouped convolution, the filters are separated into different groups. Each group is responsible for a conventional 2D convolutions with certain depth. 

The following examples can make this clearer.
![[grouped-convolutions-20220819073126884.png]]

Above is the illustration of grouped convolution with 2 filter groups. In each filter group, the depth of each filter is only half of the that in the nominal 2D convolutions. They are of depth Din / 2. Each filter group contains Dout /2 filters. The first filter group (red) convolves with the first half of the input layer (`[:, :, 0:Din/2]`), while the second filter group (blue) convolves with the second half of the input layer (`[:, :, Din/2:Din]`). As a result, each filter group creates Dout/2 channels. Overall, two groups create 2 x Dout/2 = Dout channels. We then stack these channels in the output layer with Dout channels.

### Advantages
- As was done in [[CNN Architectures#AlexNet|AlexNet]], you can split each grouped convolution up onto a different GPU. This isn't done often anymore, but was useful when GPUs had less memory.
- Grouped convolutions use less memory. In the previous examples, filters have $H \times W \times D_{\text{in}} \times D_{\text{out}}$ parameters in a nominal 2D convolution. Filters in a grouped convolution with 2 filter groups have ==$(H \times W \times \frac{D_{\text{in}}}{2} \times \frac{D_{\text{out}}}{2}) \times 2$== parameters. The number of parameters is reduced by half.
- Grouped Convolutions may also result in a better model. See [[ResNeXt]].
<!--SR:!2024-02-17,403,290-->