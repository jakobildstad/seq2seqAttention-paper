# Notes 
NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE (2014) Bahdanau et al.

## Context
This paper build upon the original seq2seq paper (Sutskever 2014) that explores using Recurrent Neural Networks (RNNs) to encode a token sequence $x$ into an embedding $c$, which then gets decoded into a translated token sequence $y$. 

The problem with Setskevers approach was that translation quality degraded as sentences bacame too long, as a result of the fixed length of $c$. 

Bahdanau improves the result by introducing attention to the system.

### What is an RNN?
A vanilla Neural Network (NN) sees an input with a **fixed** size, but speech, sequences, etc have variable lengths. To overcome this, **Recurrent** Neural Networks introduce a form of memory by having the last step as an input. This can be written as:

$h_t = f(x_t, h_{t-1})$

where:
- $x_t$ input at step $t$
- $h_t$ = hidden state from previous step

This ensures the final hidden state (embedding) $h_T$ encodes all information about the sequence inputed. 

You can write $c = q({h_1, ..., h_T})$.


### what is Bidirectional RNN (BiRNN)?
For a traditional RNN, each $h_t$ embeds information about $h_{t-1}$, from left to right. This means a h_t includes information about all tokens on the left in the sentence. 

BiRNN concatinates forwards and backwards RNN encodings such that each word "represents the whole sentence from its own perspective".

$h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]$

" Each annotation $h_i$ contains information about the whole input sequence
with a strong focus on the parts surrounding the $i$-th word of the
input sequence. "

### Bahdanau method in detail
**Step 1:**
First the encoder runs over entire input sequence and generates $h_1, h_2, ..., h_T$. These are computed up front and stored. 

**Step 2:**
The decoder gets initialized with $s_0$ from the encoder (e.g. the final state or learned transformation from it)

**Step 3:**
When the decoder wants to produce a translated word $y_i$, it:
 1. Uses previous decoder hidden state $s_{i-1}$
 2. Computes an energy score $e_{ij} = a(s_{i-1}, h_j)$ for all $j$, where a is an alignment model (feedforward NN) which is **jointly** trained with all the other components of the system. 
 3. then $e_{ij}$ is normalized through softmax to $\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}$
 4. Then the context $c_i$ is computed as $c_i = \sum_{j=1}^T \alpha_{ij} h_j$

**Step 4:**
Finally update the decoder with
$s_i = f(s_{i-1}, y_{i-1}, c_i)$
and use that hidden state to calculate the probability of the next output token $y_i$:
$ p(y_i \mid y_1, \dots, y_{i-1}, x) = g(y_{i-1}, s_i, c_i)$
