# Deep Learning Lab – Programs Index

This repository contains verified Python implementations for all internal exam and lab manual programs, organized by question number with headings, docstrings, algorithm steps, and theoretical explanations.

---

## Program Index

| File | Question & Heading | Dataset | Key Concepts / Architecture |
|---|---|---|---|
| [`q01_ffnn_cifar10_accuracy.py`](file:///c:/Users/vaish/Downloads/DL%20LAB/q01_ffnn_cifar10_accuracy.py) | **Q1.** FFNN for classification; evaluate using accuracy | CIFAR-10 | Dense 256, 256, 128 (ReLU) + Dense 10 (Softmax), Adam, Accuracy plot |
| [`q02_ffnn_mnist_digit_classification.py`](file:///c:/Users/vaish/Downloads/DL%20LAB/q02_ffnn_mnist_digit_classification.py) | **Q2.** FFNN for MNIST digit classification; display test accuracy and plot curves | MNIST | Dense 128, 64 (ReLU) + Dense 10 (Softmax), Categorical Cross-Entropy |
| [`q03_ffnn_activation_functions_comparison.py`](file:///c:/Users/vaish/Downloads/DL%20LAB/q03_ffnn_activation_functions_comparison.py) | **Q3.** FFNN with at least one hidden layer; compare activation functions | CIFAR-10 | Compares ReLU vs Sigmoid vs Tanh across Train/Val/Test accuracy |
| [`q04_ffnn_xavier_vs_kaiming_initialization.py`](file:///c:/Users/vaish/Downloads/DL%20LAB/q04_ffnn_xavier_vs_kaiming_initialization.py) | **Q4.** FFNN with Xavier and Kaiming initialization; plot loss curves and decide convergence | CIFAR-10 | Glorot Normal (Xavier) vs He Normal (Kaiming), SGD, minimum val loss epoch |
| [`q05_cnn_mnist_classification.py`](file:///c:/Users/vaish/Downloads/DL%20LAB/q05_cnn_mnist_classification.py) | **Q5.** CNN for MNIST digit classification; evaluate using test accuracy | MNIST | Conv2D(6) -> MaxPool -> Conv2D(16) -> MaxPool -> Dense(120, 84, 10), sample predictions |
| [`q06_cifar10_cnn_xavier_vs_kaiming_regularization.py`](file:///c:/Users/vaish/Downloads/DL%20LAB/q06_cifar10_cnn_xavier_vs_kaiming_regularization.py) | **Q6.** CIFAR-10 CNN with Xavier & Kaiming initialization + L2 and Dropout | CIFAR-10 | Conv2D(32, 64, 128), L2(0.001), Dropout(0.25, 0.5), accuracy & loss plots |
| [`q07_ffnn_weight_initializations_comparison.py`](file:///c:/Users/vaish/Downloads/DL%20LAB/q07_ffnn_weight_initializations_comparison.py) | **Q7.** Neural network with different weight initializations (Xavier, Kaiming) | CIFAR-10 | Baseline default vs Glorot Normal vs He Normal performance table (same as Q4) |
| [`q08_cnn_mnist_layers_train_evaluate.py`](file:///c:/Users/vaish/Downloads/DL%20LAB/q08_cnn_mnist_layers_train_evaluate.py) | **Q8.** CNN for MNIST with conv, pooling, flattening and fully connected layers | MNIST | Layer breakdown, `model.summary()`, test loss & test accuracy (same as Q5) |
| [`q11_simple_rnn_imdb_classification.py`](file:///c:/Users/vaish/Downloads/DL%20LAB/q11_simple_rnn_imdb_classification.py) | **Q11.** Simple RNN for sequence classification; train and evaluate | IMDb Reviews | Embedding(32) -> SimpleRNN(64) -> Dropout(0.2), Confusion Matrix, Precision, Recall, F1 |
| [`q12_simple_rnn_vs_lstm_imdb.py`](file:///c:/Users/vaish/Downloads/DL%20LAB/q12_simple_rnn_vs_lstm_imdb.py) | **Q12.** Simple RNN and LSTM on the same problem; compare accuracy and loss | IMDb Reviews | Identical Embedding(32) & 32 units, compares val accuracy & test loss |
| [`q13_rnn_vs_lstm_accuracy_explanation.py`](file:///c:/Users/vaish/Downloads/DL%20LAB/q13_rnn_vs_lstm_accuracy_explanation.py) | **Q13.** RNN and LSTM on IMDb; compare training/validation accuracy and explain differences | IMDb Reviews | Training & validation curve comparison + theoretical gate explanation (Forget, Input, Output) |
| [`q14_simple_rnn_vs_gru_time_comparison.py`](file:///c:/Users/vaish/Downloads/DL%20LAB/q14_simple_rnn_vs_gru_time_comparison.py) | **Q14.** Simple RNN and GRU; compare accuracy, loss and training time | IMDb Reviews | Training runtime timing via `time.perf_counter()`, test accuracy & loss |
| [`q15_simple_rnn_vs_gru_long_term_dependencies.py`](file:///c:/Users/vaish/Downloads/DL%20LAB/q15_simple_rnn_vs_gru_long_term_dependencies.py) | **Q15.** Simple RNN and GRU; analyze how GRU handles long-term dependencies | IMDb Reviews | Subplots (val accuracy & val loss) + Reset & Update gate mechanics analysis |
| [`q17_cnn_mnist_visualize_correct_vs_incorrect.py`](file:///c:/Users/vaish/Downloads/DL%20LAB/q17_cnn_mnist_visualize_correct_vs_incorrect.py) | **Q17.** CNN for MNIST digit recognition; visualize correct and incorrect test images | MNIST | LeNet CNN, separates predictions, plots 5 correctly classified (green) & 5 incorrectly classified (red) images |
| [`q19_cnn_mnist_learning_rate_convergence.py`](file:///c:/Users/vaish/Downloads/DL%20LAB/q19_cnn_mnist_learning_rate_convergence.py) | **Q19.** CNN for MNIST classification; study effect of changing learning rate | MNIST | Trains across 4 learning rates `[0.1, 0.01, 0.001, 0.0001]`, plots Training Loss & Validation Accuracy convergence curves |

---

## Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Any Program
Example:
```bash
python q05_cnn_mnist_classification.py
python q17_cnn_mnist_visualize_correct_vs_incorrect.py
python q19_cnn_mnist_learning_rate_convergence.py
```
