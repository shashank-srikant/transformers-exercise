import pandas as pd
import torch
import torch.nn as nn


class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.activation_fn = nn.ReLU()

        self.hidden_layer1 = nn.Linear(embedding_dim, hidden_dim)
        self.hidden_layer2 = nn.Linear(hidden_dim, hidden_dim)

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        # self.output_activation_fn = nn.CrossEntropyLoss()

    def forward(self, text):
        embedded = self.embedding(text)
        pooled_embedding = embedded.mean(dim=1)

        hidden_output1 = self.hidden_layer1(pooled_embedding)
        activated1 = self.activation_fn(hidden_output1)

        hidden_output2 = self.hidden_layer2(activated1)
        activated2 = self.activation_fn(hidden_output2)

        output = self.output_layer(activated2)

        # activated_output = self.output_activation_fn(output)
        return output


class Dataset:
    def __init__(self, split_ratio=0.8):
        self.vocabulary = set()
        self.vocab_index = {"<PAD>": 0}
        self.vocab_index_reverse = {0: "<PAD>"}
        self.split_ratio = split_ratio
        self.sequence_length = 0
        self.load_dataset()

    def _get_sequence_length(self):
        df = pd.read_csv("exercise0/sentiment_data.csv")
        max_len = 0
        for sentence in df["sentence"]:
            length = len(sentence.split())
            if length > max_len:
                max_len = length
        self.sequence_length = max_len

    @property
    def target_index(self):
        return {"positive": 0, "negative": 1, "neutral": 2}

    @property
    def target_index_reverse(self):
        return {0: "positive", 1: "negative", 2: "neutral"}

    def load_dataset(self):
        df = pd.read_csv("exercise0/sentiment_data.csv")

        # shuffle to ensure randomness
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        for sentence in df["sentence"]:
            for (
                word
            ) in sentence.split():  # very basic. It supposes eg: I've as single token
                self.vocabulary.add(str(word).strip().lower())

        for idx, token in enumerate(self.vocabulary, start=1):
            self.vocab_index[token] = idx
            self.vocab_index_reverse[idx] = token

        self.train_dataset = df[:100]
        self.test_dataset = df[100:150]
        self.val_dataset = df[150:]
        self.sequence_length = max(len(s.split()) for s in df["sentence"])

        print("Dataset Loaded")

    def create_index(self, sentence):
        tokens = sentence.split()
        token_indices = [
            self.vocab_index[str(token).strip().lower()] for token in tokens
        ]

        while len(token_indices) < self.sequence_length:
            token_indices.append(self.vocab_index["<PAD>"])

        return torch.tensor(token_indices)

    def batch_index(self, sentences):
        return torch.stack([self.create_index(sentence) for sentence in sentences])


def train():
    dataset = Dataset()
    num_epochs = 100
    batch_size = 16
    lr = 0.01
    total_samples = 0
    total_val_loss = 0

    model = SentimentClassifier(
        vocab_size=len(dataset.vocab_index),
        embedding_dim=50,
        hidden_dim=100,
        output_dim=len(dataset.target_index),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        # training phase
        for iteration in range(0, len(dataset.train_dataset), batch_size):
            batch_sentences = dataset.train_dataset["sentence"][
                iteration : iteration + batch_size
            ]
            batch_labels = dataset.train_dataset["label"][
                iteration : iteration + batch_size
            ]

            batch_indices = dataset.batch_index(batch_sentences)
            batch_labels = torch.tensor(
                [dataset.target_index[label] for label in batch_labels]
            )

            optimizer.zero_grad()
            logits = model.forward(batch_indices)
            loss = criterion(logits, batch_labels)

            loss.backward()
            optimizer.step()  # weight update

        print(
            f"Epoch: {epoch},Loss: {total_val_loss / total_samples if total_samples > 0 else 0:.4f}"
        )
        # validation phase
        total_correct = 0

        with torch.no_grad():
            for iteration in range(0, len(dataset.val_dataset), batch_size):
                batch_sentences = dataset.val_dataset["sentence"][
                    iteration : iteration + batch_size
                ]
                batch_labels = dataset.val_dataset["label"][
                    iteration : iteration + batch_size
                ]

                batch_indices = dataset.batch_index(batch_sentences)
                batch_labels = torch.tensor(
                    [dataset.target_index[label] for label in batch_labels]
                )

                logits = model.forward(batch_indices)
                predictions = torch.argmax(logits, dim=1)
                total_correct += (predictions == batch_labels).sum().item()
                total_samples += batch_labels.size(0)

                loss = criterion(logits, batch_labels)

                total_val_loss += loss.item() * len(batch_labels)

                print(
                    f"Accuracy: {total_correct / total_samples if total_samples > 0 else 0:.4f}"
                )
                print(f"Validation Iteration: {iteration}, Loss: {loss.item()}")


if __name__ == "__main__":
    train()


"""
Vocabulary: {"i": 0, "love": 1, "this": 2, "movie": 3, "hate": 4}
Sentence: I love this  movie
tokenization: [0, 1, 2, 3]
---
vocab_size = 5
embedding_dim = 3
---
The values are randomly initialized. Each number represents a dimension and is learnt during the training process.
Index 0 = ("i") = [0.1, 0.2, 0.3]
Index 1 = ("love") = [0.4, 0.5, 0.6]
Index 2 = ("this") = [0.7, 0.8, 0.9]
Index 3 = ("movie") = [0.2, 0.3, 0.4]
Index 4 = ("hate") = [0.5, 0.6, 0.7]
---
For "I love this movie", the tokenization is [0,1,2,3].
4 tokens, each token represented by a vector of 3 dimensions. So, that's a 4x3 matrix.
---
Now, with batching for 2 sentences:
"I love this movie" = [0, 1, 2, 3]
"I hate this movie" = [0, 4, 2, 3]

The batches are stacked, so we have a 2x4 matrix. (batch_size * sequence_length)
After embedding lookup, we get 2x4x3 matrix. (batch_size * sequence_length * embedding_dim)
because, each of 4 sequences in both sentences in represented by a vector of 3 dimensions as:
[
[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.2, 0.3, 0.4]],
[[0.1, 0.2, 0.3], [0.5, 0.6, 0.7], [0.7, 0.8, 0.9], [0.2, 0.3, 0.4]]
]

Q1: Why ReLU activation function is used in the hidden layers?
- ReLU (Rectified Linear Unit) is computationally efficient and introduces non-linearity, allowing the model to learn complex patterns in the data.
- It helps to mitigate the vanishing gradient problem, which can occur with other activation functions like sigmoid or tanh,
    by allowing gradients to flow through the network more effectively during training.

Q2: Why sigmoid activation function is not used in the output layer?
- Sigmoid is suitable for binary classification, as output is between 0 and 1.
- It doesn't guarantee that sum of outputs will be 1, which is required for multi-class classification.
- Softmax is used in multi-class classification as it converts output scores into probabilities that sum to 1.
- CrossEnropyLoss uses softmax internally.
- CrossEntropyLoss uses negative log-likelihood loss.
- eg, for 3 classes, if outputs from output layer are [1, 0, 0]:
    - After softmax probabilities could be [0.7, 0.2, 0.1] which means model is predicting class 0 with 70% confidence.
    - After cross-entropy loss, the loss would be -log(0.7) = 0.3567, which is low, indicating a good prediction.
    - A wrong prediction like [0.1, 0.8, 0.1] would yield a higher loss of -log(0.1) = 2.3026, that will help model learn faster.
- We can use softmax then nn.NLLLoss, CrossEnropyLoss is preferred to avoid numerical instability issues.
    (like, softmax: which includes e^x can include very small probabilities that can lead to overflow/underflow)


Q3: WHy pooling when passing embeddings to the hidden layers?
- In case of batch, the size of matrix doesn't match with the expected input size of hidden layers.
- Eg: In our above case, 2x4x3 isn't accepted by hidden layers (embedding_dim, hidden_dim).

---
Training Data
Q1: Input data can have sentences of varying lengths. How do we handle that?
- We can pad the sentences, eg: "I love this movie" becomes [1,2,3,4] and "good" becomes [5,0,0,0].
- Later when mean pooling, padded values should be ignored.

Q2:

"""
