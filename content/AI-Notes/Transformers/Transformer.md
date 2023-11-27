---
tags: [flashcards]
source:
summary:
---

**Resources**:
For an optimized sequence-to-sequence implementation see [here](https://github.com/tunz/transformer-pytorch)
For an implementation of transformers that matches the paper see [here](https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py)

**General**:
Transformers have been called the "ImageNet Moment for Natural Language Processing." You can pre-train a giant transformer model (many blocks) on a lot of internet text. You can the fine-tune the transformer for your specific NLP task.

Fundamentally, a transformer uses blocks with two basic operations. First, is an [[Attention]] operation for modeling inter-element relations. Second, is a multi-layer perceptron ([[Linear|mlp]]), which models relations within an element. Intertwining these operations with normalization and residual connections allows transformers to generalize to a wide variety of tasks. Source: [[Multiscale Vision Transformers|MViT]].

> [!NOTE] The example implementation is for a sequence-to-sequence transformer (where position matters).

# Encoder-decoder architecture
![[annotated-transformer-architecture.png]]
Like earlier seq2seq models, the original Transformer model used an **encoder-decoder** architecture. The encoder consists of encoding layers that process the input iteratively one layer after another, while the decoder consists of decoding layers that do the same thing to the encoder's output.

The function of each encoder layer is to generate encodings that contain information about which parts of the inputs are relevant to each other. It passes its encodings to the next encoder layer as inputs. Each decoder layer does the opposite, taking all the encodings and using their incorporated contextual information to generate an output sequence. To achieve this, each encoder and decoder layer makes use of an attention mechanism.

For each input, attention weighs the relevance of every other input and draws from them to produce the output. Each decoder layer has an additional attention mechanism that draws information from the outputs of previous decoders, before the decoder layer draws information from the encodings from the ==final layer== of the encoder.
<!--SR:!2027-01-22,1364,330-->

Both the encoder and decoder layers have a [feed-forward neural network](https://en.wikipedia.org/wiki/Feedforward_neural_network "Feedforward neural network") for additional processing of the outputs and contain residual connections and layer normalization steps.

# Encoder
**From the paper**: The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position- wise fully connected feed-forward network. We employ a residual connection [11] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is $\operatorname{LayerNorm}(x+\text { Sublayer }(x))$, where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{\text {model }}=512$.
![[AI-Notes/Transformers/transformer-srcs/transformer-encoder.png]]

- First you get a sequence of inputs $X_i$
- You then pass the inputs into a [[Attention#Self-Attention Layer|self-attention layer]]. This makes all the vectors interact with each other to generate outputs that depend on a weighted combination of the transformed inputs. **This is the only time there is interaction between vectors.**
    -  In a self-attention layer all of the keys, values and queries come from the ==same== place, in this case, the output of the ==previous layer in the encoder==. Each position in the encoder can attend to all positions in the previous layer of the encoder.
- You add a residual connection from the original inputs.
- You then normalize the layers. This operates on each output vector **independently**.
- You then pass the normalized layers through an MLP (multi-layer perceptron via a [[Pointwise Convolution]]). This will again operate **independently** on each of the vectors.
- You add another residual connection
- You normalize the layers again. This operates **independently** per vector.
- You give the outputs.
<!--SR:!2024-02-23,557,330!2028-09-11,1855,350-->

**Note**: the encoder stacks multiple encoder layers on top of each other, so one encoder receives the output from the previous encoder. Only the last encoder output is given to decoder as input.
```python
class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.src_word_emb(src_seq)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list

        # Note: only the last layer of the encoder is returned.
        return enc_output,
```

The following diagram makes it clearer that the encoder output from one layer is passed to next the layer. Only the final output 

# Decoder
**From the paper**: The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs ==multi-head attention over the output of the encoder stack==. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the ==known outputs at positions less than i==.
![[transformer-decoder.png]]
- In "encoder-decoder attention" layer (labelled "Multi-Head Attention" above), the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence.
- Self-attention layers in the decoder (labelled "Masked Multi-Head Attention) allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections.
```python
class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,
```
<!--SR:!2024-05-30,613,308!2026-01-27,1113,348-->

# Embeddings and Softmax
- Use learned embeddings (aka applying a learned weight matrix as a linear transform) to convert the input and output tokens to vectors of dimension $d_{\text {model }}$.
- Use a learned linear transformation to convert the decoder output to predicted next-token probabilities.
- The original paper shared the same weight matrix between the two embedding layers and the pre-softmax linear transformation.

# Positional Encodings
- Transformers contain no recurrence and no convolution, so in order to make use of the order of the sequence, positional encodings are added to the input embeddings at the bottom of the encoder and decoder stacks.
- The positional encodings have the same dimension $d_{\text {model }}$ as the embeddings, so that the two can be summed.
- Most modern approaches use learned positional embeddings, but the original paper used a sinusoid.

# Usage

The transformable is highly scalable and parallelizable. You get a set of vectors $X$ and output a set of vectors $Y$ (the number of vectors in $X$ and $Y$ are the same, but they can have a different dimensionality). The only time the input vectors interact is through self-attention.

> [!note]
> The greater the number of transformer layers, width of the layer, and heads ⇒ the better performance you seem to get. The only constraint on getting better performance is how much computation you can get.
A **Transformer** is a sequence of transformer blocks. You only need a few hyperparameters to specify the architecture of a transformer:
- The number of transformer blocks.
- The dimension of the query vectors inside the blocks.
- The number of heads in the [[Attention#Multihead Self-Attention Layer|multihead self-attention layers]].

Transformer models are very popular in NLP for training on large text datasets (try toi predict the next letter given the previous letter)  and then using them to generate new text. It blows past [[RNN, LSTM, Captioning|RNN]] models. You can also [[fine-tune]] the transformer on your own NLP task. There is no supervision of humans needed (no human labelling of the text). You can just keep training on larger datasets (kicked off a race between labs to train bigger transformers on bigger datasets).

![[scaling-up-transformers.png]]
- The number of layers refers to the number of transformer blocks
- The width refer to the dimension of the vector that stores the input (input embedding).

# Transformer Implementation
[Source](https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py)
```python
class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj'):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight


    def forward(self, src_seq, trg_seq):

        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)

        # Note that only the final output of the encoder is passed to the decoder
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_word_prj(dec_output)
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5

        return seq_logit.view(-1, seq_logit.size(2))
```
**Note**: this implementation calculates the encoder embedding for each step. For a sequence-to-sequence model, this is inefficient because you will ==recompute the encoder embedding for each output step== during inference (while training all the computation can be done in parallel using [[Teacher Forcing]]). For an efficient implementation for sequence-to-sequence transformers see [[Optimized Autoregressive Transformer]].
<!--SR:!2024-07-31,637,310-->

# Sources
[Good article with nice diagrams](https://towardsdatascience.com/transformer-attention-is-all-you-need-1e455701fdd9)
[Wikipedia article](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model))
