# first model, what about second model

# starting with logistical regressin just like anything else
# starting with p1 baseline implementation
from tqdm import tqdm
import numpy as np
from numpy import ndarray
from scipy import sparse
from sklearn.model_selection import train_test_split
import pdb

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from p1_utils import *


def sigmoid(val: ndarray) -> ndarray:
    return 1.0 / (1.0 + np.exp(-1 * val))


# log likelihood
def ll(b: ndarray, y: ndarray, x) -> float:
    acc = 0
    for i, y_i in enumerate(y):
        x_i = x[i]
        btx_i = x_i.dot(b)  # prediction
        acc += y_i * (btx_i) - np.log(1 + np.exp(btx_i))
    return acc


# collapsing for loop is over 50x faster
def ll_fast(b: ndarray, y: ndarray, x) -> float:
    btx = x * b
    return float(np.sum(btx * y - np.log(1 + np.exp(btx))))


def grad_ll(b: ndarray, y_i: ndarray, x_i: ndarray) -> ndarray:
    return (sigmoid(x_i.dot(b)) - y_i) * x_i


# x is sparse, hence untyped
def logistic_regression(
    x, y: ndarray, learning_rate: float, num_step: int, loss_capture=1000
) -> tuple[ndarray, ndarray]:
    row_len = x.shape[1]

    b = np.zeros(row_len)
    lls = np.zeros((num_step // loss_capture) + 1)
    for i in tqdm(range(num_step)):
        if i % loss_capture == 0:
            lls[(i // loss_capture)] = ll_fast(b, y, x)
        rand_index = i % len(y)
        row = x[rand_index]
        label = y[rand_index]
        b = b - learning_rate * grad_ll(b, label, row)
    lls[-1] = ll_fast(b, y, x)

    return b, lls


def predict(text: str, b: ndarray, vocab_index: dict[str, int]):
    x = input_to_vector(text, vocab_index)
    y_hat = sigmoid(b.dot(x))
    return y_hat > 0.5


# predict continuous
def predict_cont(text: str, b: ndarray, vocab_index: dict[str, int]):
    x = input_to_vector(text, vocab_index)
    y_hat = b.dot(x)

    # clamp between 0 and 1
    y_hat = np.clip(y_hat, 0, 1)
    return y_hat


def input_to_vector(text: str, vocab_index: dict[str, int]) -> ndarray:
    val = np.zeros(len(vocab_index) + 1)
    val[-1] = 1  # bias
    for word in better_tokenize(text):
        if word in vocab_index:
            val[vocab_index[word]] += 1
    return val


def main():
    print("before data read")
    df: pd.dataframe = pd.read_csv("generated_post_classification.csv")
    train_df, test_df = train_test_split(df, test_size=0.2)

    print(f"train size:{len(train_df)}, test size:{len(test_df)}")
    print(f"train data: {train_df} test data: {test_df}")

    print("before vocab generation")
    vocabulary, vocab_index = derive_vocab(train_df, key="assignment")

    print("before df split")
    docs = train_df["assignment"]
    labels = train_df["error"]

    print("before label map generation")

    label_ids = np.array(train_df["error"]).astype(float)
    print(label_ids)

    # next step: create a sparse d x v matrix
    # note a bias term will be eventually needed

    print("before sparse mat population")

    mat_csr = sparse_dv_mat(train_df, vocab_index, key="assignment").tocsr()
    # coo is probably the correct way to do it

    print("before logistic regression")

    epochs = 10

    b, lls = logistic_regression(
        mat_csr, label_ids, 1e-4, epochs * len(label_ids), loss_capture=1
    )
    sns.lineplot(x=range(len(lls)), y=lls)
    # set y axis font size
    plt.gca().axes.yaxis.label.set_size(14)
    # set x axis font size
    plt.gca().axes.xaxis.label.set_size(14)

    # set y axis title
    plt.gca().set_ylabel("log likelihood")
    # set x axis title
    plt.gca().set_xlabel("iteration")

    # set title
    plt.title("Logistic Regression Loss over Iterations")
    # set font size
    plt.gca().axes.title.set_size(14)

    plt.savefig("second_model_loss.png", dpi=300)

    # print("load test data")
    # test_df = pd.read_csv("generated_with_labels.csv")

    print("run predictions")
    # this part is now different

    # for random baseline, calculate the average error
    avg_error = sum(train_df["error"]) / len(train_df["error"])

    mse_baseline = np.mean((test_df["error"] - avg_error) ** 2)

    preds = np.zeros(test_df["assignment"].count(), dtype=float)
    i = 0
    for idx in tqdm(test_df["assignment"].keys()):
        text = test_df["assignment"][idx]
        preds[i] = predict_cont(text, b, vocab_index)
        i += 1

    mse_model = np.mean((test_df["error"] - preds) ** 2)

    print(f"mse_baseline:{mse_baseline}")
    print(f"mse_model:{mse_model}")


if __name__ == "__main__":
    main()
