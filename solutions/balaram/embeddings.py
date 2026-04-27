import torch
from torch import nn


class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.we_linear = nn.Linear(embedding_dim, hidden_dim)
        self.wh_linear = nn.Linear(hidden_dim, hidden_dim)
        # both linear layers above output, hidden_dim size, so they can be added.

        self.activation = nn.Tanh()  # -> (batch_size, hidden_dim)

    def forward(self, text):
        batch_size = text.size(0)
        sequence_len = text.size(1)
        h = torch.zeros(batch_size, self.hidden_dim)

        for t in range(sequence_len):
            tokens_t = text[:, t]
            e_t = self.embedding(tokens_t)  # size will be (batch_size, embedding_dim)

            h = self.activation(self.we_linear(e_t) + self.wh_linear(h))

        return h


"""
Source: https://huggingface.co/spaces/hesamation/primer-llm-embedding
Approach 1: TF-IDF
- Term Frequency (TF): Measures how frequently a term appears in a document.
tf(t,d) = (Number of times term t appears in document d) / (Total number of terms in document d)
- Inverse Document Frequency (IDF): Measures how important a term is.
idf(t) = log_e(Total number of documents / Number of documents with term t in it)
    - It gives higher weight to terms that appear less commonly across documents. So, filler words have low IDF score.

Sparse/Frequency based Representation:
1. One-hot encoding
2. TF-IDF
3. Bag of words

Word Embedding Learning:
1. Word2Vec
    CBOW: Predict the target word from context words.
    Skip-gram: Predict context words from the target word.

Sentence Embedding Representation Approaches:
1. Mean Pooling (that was used in previous sentiment example)
2. RNN
3. Attention


Words -> Word Embeddings (word2vec, one-hot, learned embeddings) -> Sentence Repr (Mean pooling, RNN, Attention) -> Downstream task


RNN:
e0, h0 -> RNN -> h1
                e1, h1 -> RNN -> h2
                                 e2, h2 -> RNN -> h3
Where en is embedding of nth token in the sentence.
h0 is randomly initialized hidden state.
hn is the hidden state after processing token n.


Example Tracing:
S1: I am not happy
Embeddings: e1, e2, e3, e4
Hidden state: h0(initial)

h1 = RNN(e1, h0) -> h1 (h1, carries info about "I")
h2 = RNN(e2, h1) -> h2 (h2, carrries info about "I am")
h3 = RNN(e3, h2) -> h3 (h3, carries ifnro about "I am not")
h4 = RNN(e4, h3) -> h4 (h4, carries info about "I am not happy")

What would the RNN function look like?
In simple cases,
h_i = activation(W_h * h_{i-1} + W_e * e_i + b)

Where:
    - W_h is the weight matrix for the hidden state
    - W_e is the weight matrix for the input embedding
    - h_{i-1} is the previous hidden state
    - b is the bias term
    - activation is a non-linear function like tanh or ReLU

Formula Understanding:
1. At each step, want to combine, what I remember so far (h_{i-1}) with what I'm currently seeing (e_i).
2. W_h transforms previous hidden states (with W_h * h_{i-1}), it get's updated after each step. So, model learns to emphasize certain information from the past.
3. Why W_e * e_i? W_e transforms current word embedding. Model learns what to extract from the word. Eg Words like "not", "love" activate strongly.
4. Added to join both signals into one. While each being learnt separately.
5. Activation function for non-linearity, but why tanh?
    With others, let's say ReLU, outputs values [0, inf). If at each step, value grow even slightly, it can explode.
    With tanh ([-1, 1]), the values remain in bounded range. Although, vanishing gradients (approaching 0) is a potential problem.


Vanishing Gradient:
- The information from earlier tokens get diluted after many stes.


GRU (Gated Recurrent Unit):
- Use gates (reset and update) to control how much of past information to keep and forget.
- From basic formula, gating only embedding state would be useful to decided if current word is important or not.
  But, if we don't gate hidden state, there are cases when we need to forget the entire past context.
  Eg: I loved the video work, but the movie was terrible.
  - Hidden state after "but", should be reset.
- So, GRU has gates for both hidden state and embedding state. (reset gate and update gate)
- Formula:
    r = sigmoid(W_r * [h_{i-1}, e_i] + b_r)  # concatenated
    z = sigmoid(W_z * [h_{i-1}, e_i] + b_z)
    candidate_hidden_state (h_candidate) = tanh(W_h * [r*h_{i-1}, e_i] + b_h)
    final_state = h_i = z* h_{i-1} + (1-z) * h_candidate


    if h_{i-1} = [0.5, 0.5], e_i = [0.2, 0.8]
    [h_{i-1}, e_i] = [0.5, 0.5, 0.2, 0.8]



-------------------------------------------   
# With Mean Pooling:  
    embedded = self.embedding(text)
    pooled_embedding = embedded.mean(dim=1)

    if, batch_size=16, seq_len=10, embedding_dim=50
    embedded.shape = (16, 10, 50) and
    pooled_embedding.shape = (16, 50) 

    
# With RNN:
self.encoder = RNNEncoder(vocab_size, embedding_dim, hidden_dim)
embedding = self.encoder(text)
if, batch_size=16, seq_len=10, embedding_dim=50, hidden_dim=50
embedding.shape = (16, 50)

    
--------------------------------------------------
Q: What problem does an RNN solve that mean pooling doesn't ?
A: - Mean pooling averages the embeddings of all tokens in a sequence, which can lead to loss of important contextual information.
   - RNNs, process words sequentially, maintaining a hidden state that carries information forward.
   - eg:, With "I am happy" v/s "I am not happy".
    -> Mean Pooling: Happy and not happy contribute their embeddings equally.The "not" doesn't modify "happy", but is averaged together with everything else.
    -> RNN: With sequential processing, happy is processed in the context of having already seen "I am not".

"""
