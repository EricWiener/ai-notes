---
tags: [flashcards]
source: https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be
summary: similar to LSTMs, but use a simpler structure.
---
[[Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation, Kyunghyun Cho et al., 2014.pdf]]

The GRU is the newer generation of Recurrent Neural networks and is pretty similar to an LSTM. GRU’s got rid of the ==cell state== and used the ==hidden state== to transfer information. It also only has two gates, a ==reset gate and update== gate (vs. the LSTM's 4 gates).
<!--SR:!2023-12-24,84,250!2028-04-04,1724,350!2024-01-02,44,251-->

### RNN and LSTM:
[[RNN, LSTM, Captioning|RNN]]:
![[multiple-gradient-flow.png]]
$$h_{t}=\tanh \left(W\left(\begin{array}{c}
h_{t-1} \\
x_{t}
\end{array}\right)+b_{h}\right)$$
[[RNN, LSTM, Captioning|LSTM]]:
![[lstm-info-flow.png|500]]
![[lstm-equations.png|300]]

### GRU:
![[gru-20220314090842202.png|500]]


 ![[gru-equations.png|500]]
- The reset gate $r_t$ decides how much of the previous hidden state ($h_{t-1}$) is ignored when computing the ==current memory context== ($\tilde{h}_{t}$). The current memory context takes into account the previous hidden states and the current input $x$.
    - It is computed using weight matrices $W_{xr}$ and $W_{hr}$ and a bias $b_r$.
    - It depends on the input to the layer and the previous hidden state.
    - When the resetgate is close to 0, the hidden state is forced to ignore the previous hidden state and reset with the current input. This allows the hidden state to drop any information that is found to be irrelevant later in the future (allowing a more compact representation).
- The update gate $z_t$ controls how much information from the current memory context ($\tilde{h}_{t}$) will carry over to the hidden state that is output at this step ($h_t$).
    - This works similarly to the memory cell in the LSTM and helps remember long-term information.
- $\tilde{h}_{t}$ (denoted $h'_t$ in the diagram) is the current memory context. This uses the reset gate to store the relevant relevant information from the past. This is a temporary value and is used to calculate the real hidden state $h_t$.
- $h_{t}$ is the hidden state that will be passed to future layers of the network.
<!--SR:!2024-07-30,597,338-->

**GRU Analysis**:
- As each hidden unit has separate reset and update gates, each hidden unit will learn to capture dependencies over different time scales.
- Units that capture short-term dependnecies will have reset gates that are frequently active.
- Units that capture longer-term dependencies will have update gates that are mostly active.
- With LSTMs, you passed the cell-state from layer to layer. You would still use the previous hidden layer as input (it was stacked with the input $x$ and multiplied by $W$ and then split into the corresponding gates). Now, you are using the hidden layers for the following states only instead of keeping an additional cell state.