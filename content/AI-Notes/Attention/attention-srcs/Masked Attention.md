---
source: https://stackoverflow.com/a/59713254/6942666
---

> [!note] How to understand masked attention in transformers
> 

I had the very same question after reading the [Transformer paper][1]. I found no complete and detailed answer to the question in the Internet so I'll try to explain my understanding of Masked Multi-Head Attention.

The short answer is - we need masking to make the training parallel. And the parallelization is good as it allows the model to train faster.

Here's an example explaining the idea. Let's say we train to translate "I love you" to German. The encoder works in parallel mode - it can produce vector representation of the input sequence ("I love you") within a constant number of steps (i.e. the number of steps doesn't depend on the length of the input sequence).

Let's say the encoder produces the numbers `11, 12, 13` as the vector representations of the input sequence. In reality these vectors will be much longer but for simplicity we use the short ones. Also for simplicity we ignore the service tokens, like `<start>` - beginning of the sequence, `<end>` - end of the sequence and others.

During the training we know that the translation should be "Ich liebe dich" (we always know the expected output during the training). Let's say the expected vector representations of the "Ich liebe dich" words are `21, 22, 23`.

If we make the decoder training in sequential mode, it'll look like the training of the Recurrent Neural Network. The following sequential steps will be performed:

 - Sequential operation #1. Input: `11, 12, 13`.
   - Trying to predict `21`.
   - The predicted output won't be exactly `21`, let's say it'll be `21.1`.
 - Sequential operation #2. Input: `11, 12, 13`, and also `21.1` as the previous output.
   - Trying to predict `22`.
   - The predicted output won't be exactly `22`, let's say it'll be `22.3`.
 - Sequential operation #3. Input `11, 12, 13`, and also `22.3` as the previous output.
   - Trying to predict `23`.
   - The predicted output won't be exactly `23`, let's say it'll be `23.5`.

This means we'll need to make 3 sequential operations (in general case - a sequential operation per each input). Also we'll have an accumulating error on each next iteration. Also we don't use attention as we only look to a single previous output.

As we actually know the expected outputs we can adjust the process and make it parallel. There's no need to wait for the previous step output (note: this is [[Teacher Forcing]]).

 - Parallel operation `A`. Inputs: `11, 12, 13`.
   - Trying to predict `21`.
 - Parallel operation `B`. Inputs: `11, 12, 13`, and also `21`.
   - Trying to predict `22`.
 - Parallel operation `C`. Inputs: `11, 12, 13`, and also `21, 22`.
   - Trying to predict `23`.

This algorithm can be executed in parallel and also it doesn't accumulate the error. And this algorithm uses attention (i.e. looks to all previous inputs) thus has more information about the context to consider while making the prediction.

And here is where we need the masking. The training algorithm knows the entire expected output (`21, 22, 23`). It hides (masks) a part of this known output sequence for each of the parallel operations.

 - When it executes `A` - it hides (masks) the entire output.
 - When it executes `B` - it hides 2nd and 3rd outputs.
 - When it executes `C` - it hides 3rd output.

Masking itself is implemented as the following (from the [original paper][1]): 

> We implement this inside of scaled dot-product attention by masking
> out (setting to −∞) all values in the input of the softmax which
> correspond to illegal connections

Note: during the inference (not training) the decoder works in the sequential (not parallel) mode as it doesn't know the output sequence initially. But it's different from RNN approach as Transformer inference still uses self-attention and looks at all previous outputs (but not only the very previous one).

Note 2: I've seen in some materials that masking can be used differently for non-translation applications. For example, for language modeling the masking can be used to hide some words from the input sentence and the model will try to predict them during the training using other, non-masked words (i.e. learn to understand the context).


  [1]: https://arxiv.org/abs/1706.03762