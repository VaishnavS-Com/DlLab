import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense

num_words, maxlen = 10000, 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

layers = {"SimpleRNN": SimpleRNN, "LSTM": LSTM}
histories = {}

for name, layer in layers.items():
    model = Sequential([
        Embedding(num_words, 32),
        layer(32),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    histories[name] = model.fit(x_train, y_train, epochs=5, batch_size=64,
                                validation_split=0.2, verbose=0)
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{name}: Test Accuracy={acc:.4f}")

plt.figure(figsize=(8, 5))
for name, h in histories.items():
    plt.plot(h.history["accuracy"], label=f"{name} Train")
    plt.plot(h.history["val_accuracy"], "--", label=f"{name} Val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy: SimpleRNN and LSTM")
plt.legend()
plt.grid(True)
plt.show()

# Explanation: SimpleRNN has a single hidden state, and gradients shrink or
# explode over long sequences (vanishing/exploding gradient), so it forgets
# early words. LSTM adds a cell state controlled by input, forget and output
# gates, which decide what to store, discard and output. This lets it retain
# long-term dependencies and usually gives more stable, higher accuracy on
# long reviews.

