import torch
import torch.nn.functional as F

corpus = "The quick brown fox jumps over the lazy dog".lower()
corpus_list = corpus.split()


def data_preparation():

    window_size = 2
    context_target_pairs = []

    for index in range(window_size, len(corpus_list) - window_size):
        context = (
            corpus_list[index - window_size : index]
            + corpus_list[index + 1 : index + window_size + 1]
        )
        target = corpus_list[index]
        print(f"Context: {context} --> Target: {target}")
        context_target_pairs.append((context, target))
    return context_target_pairs


def create_vocabulary():
    vocab = set(corpus_list)
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    index_to_word = {idx: word for word, idx in word_to_index.items()}

    return word_to_index, index_to_word


def one_hot_encode(word, word_to_index):
    one_hot = torch.zeros(len(word_to_index))
    one_hot[word_to_index[word]] = 1.0
    return one_hot


class CBOW:
    def __init__(self, vocab_size, embedding_dim):
        self.W1 = torch.randn(vocab_size, embedding_dim, requires_grad=True)
        self.W2 = torch.randn(embedding_dim, vocab_size, requires_grad=True)

    def forward(self, context_words):
        h = context_words.mean(dim=0) @ self.W1
        logits = h @ self.W2
        return logits


def train(model, context_target_pairs, word_to_index, epochs=100, lr=0.01):
    for epoch in range(epochs):
        total_loss = 0.0

        for context, target in context_target_pairs:
            context_words = torch.stack(
                [one_hot_encode(w, word_to_index) for w in context]
            )
            target_idx = torch.tensor(word_to_index[target])

            logits = model.forward(context_words)

            loss = F.cross_entropy(
                logits.unsqueeze(0), target_idx.unsqueeze(0)
            )  # cross entropy expects (batch,vocab) and (batch,), so added a batch dimension.

            loss.backward()

            with torch.no_grad():
                model.W1 -= lr * model.W1.grad
                model.W2 -= lr * model.W2.grad

                model.W1.grad.zero_()
                model.W2.grad.zero_()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d} ===== Loss: {total_loss:.4f}")


def predict(model, context_words, word_to_index, index_to_word):
    with torch.no_grad():
        context_tensor = torch.stack(
            [one_hot_encode(w, word_to_index) for w in context_words]
        )

        logits = model.forward(context_tensor)

        probs = torch.softmax(logits, dim=0)
        predicted_idx = torch.argmax(probs).item()

    return index_to_word[predicted_idx], probs.tolist()


def main():
    context_target_pairs = data_preparation()
    print(f"\nContext Target pairs: {context_target_pairs}")

    word_to_index, index_to_word = create_vocabulary()
    print(f"\nWord to Index Dictionary: {word_to_index}")

    context_one_hot = [one_hot_encode(word, word_to_index) for word in ["the", "quick"]]
    print(f"One-Hot Encoding: {context_one_hot}")

    model = CBOW(vocab_size=len(word_to_index), embedding_dim=5)

    train(
        model=model,
        context_target_pairs=context_target_pairs,
        word_to_index=word_to_index,
    )

    print("-------------- Prediction ----------------")
    test_context = ["the", "quick", "fox", "jumps"]

    predicted_word, _ = predict(model, test_context, word_to_index, index_to_word)

    print(f"Target: {predicted_word} \nProbability List: {_}")


if __name__ == "__main__":
    main()

"""
CBOW predicts a target word from it's surrounding context words. 
Example: "the cat sits on the mat", window_size = 2, (two words left, two words right)
Context = ["the", "cat", "on", "the"] -> word between "cat" and "on" is guessed i.e sits


CBOW uses the average of embeddings of context words to predict the target word using a probability distribution. 


Steps: 
1. Data Preparation
2. Generate Context Target Pairs
3. One-Hot Embedding
4. Embedding Layer
5. Context Aggregation
6. Prediction
7. Loss Calculation and Optimization
8. Repeat for all Pairs.


Embedding Layer: embedding matrix W of size Vxd where V is the vocabulary size and d is the embedding dimension. 
Context Aggregation: Embedding of all context words are averaged to compute the context vector. 


Example of CBOW: 
Input: "I love machine learning".
    target_word: machine
    context_words: ["I", "love", "learning"]

One Hot Encoding: 
    Vocabulary = ["i", "love", "machine", "learning", "ai]
    One-hot vectors: 
        "I" -> [1,0,0,0,0][1,0,0,0,0][1,0,0,0,0]
        "love" -> [0,1,0,0,0][0,1,0,0,0][0,1,0,0,0]
        "learning" -> [0,0,0,1,0][0,0,0,1,0][0,0,0,1,0]

Embedding Layer: 
    embedding_dim(d) = 3
    embedding_matrx (W)= 
                    [
                    0.1     0.2     0.3
                    0.4     0.5     0.6
                    0.7     0.8     0.9
                    0.2     0.3     0.4
                    0.5     0.6     0.7
                    ]

Embeddings: 
    "i" -> [0.1, 0.2, 0.3]
    "love"-> [0.4, 0.5, 0.6]
    "learning" -> [0.2, 0.3, 0.4]


Aggregation: 
    Average the embeddings: 
        h = ([0.1, 0.2, 0.3] + [0.4, 0.5, 0.6] + [0.2, 0.3, 0.4]) / 3 = [0.233, 0.333, 0.433]


Output Layer: 
    compute logits, apply softmax, and predict the target word. 

"""
