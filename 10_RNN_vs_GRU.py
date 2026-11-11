import time
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, GRU, Dense

num_words, maxlen = 10000, 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

layers = {"SimpleRNN": SimpleRNN, "GRU": GRU}
histories, times = {}, {}

for name, layer in layers.items():
    model = Sequential([
        Embedding(num_words, 32),
        layer(32),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    start = time.perf_counter()
    histories[name] = model.fit(x_train, y_train, epochs=5, batch_size=64,
                                validation_split=0.2, verbose=0)
    times[name] = time.perf_counter() - start
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{name}: Test Loss={loss:.4f}, Test Accuracy={acc:.4f}")

for name, sec in times.items():
    print(f"{name} training time: {sec:.2f} seconds")

for name, h in histories.items():
    plt.plot(h.history["val_accuracy"], label=name)
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("SimpleRNN vs GRU")
plt.legend()
plt.grid(True)
plt.show()

