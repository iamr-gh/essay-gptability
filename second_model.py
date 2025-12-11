# first model, what about second model

# starting with logistical regression just like anything else
# starting with p1 baseline implementation
from tqdm import tqdm
import numpy as np
from numpy import ndarray
from scipy import sparse
from sklearn.model_selection import train_test_split
from sys import argv

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
    # Set global plotting parameters for poster quality
    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.labelsize": 28,
            "axes.titlesize": 32,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 22,
            "figure.figsize": (12, 8),
            "lines.linewidth": 3,
            "axes.linewidth": 2,
            "xtick.major.width": 2,
            "ytick.major.width": 2,
            "xtick.major.size": 8,
            "ytick.major.size": 8,
        }
    )

    sns.set_style("dark")
    sns.set_context("poster")

    filename = argv[1]
    print("before data read")
    df: pd.DataFrame = pd.read_csv(filename)

    train_df, test_df = train_test_split(df, test_size=0.2)

    print(f"train size:{len(train_df)}, test size:{len(test_df)}")
    print(f"train data: {train_df} test data: {test_df}")

    print("before vocab generation")
    vocabulary, vocab_index = derive_vocab(train_df, key="user_prompt")

    print("before df split")
    docs = train_df["user_prompt"]
    labels = train_df["error"]

    print("before label map generation")

    label_ids = np.array(train_df["error"]).astype(float)
    print(label_ids)

    print("before sparse mat population")

    mat_csr = sparse_dv_mat(train_df, vocab_index, key="user_prompt").tocsr()

    print("before logistic regression")

    epochs = 10

    b, lls = logistic_regression(
        mat_csr, label_ids, 1e-5, epochs * len(label_ids), loss_capture=1
    )

    # Plot 1: Log Likelihood over Iterations
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.plot(range(len(lls)), lls, linewidth=3.5, color="#2E86AB")
    ax.set_ylabel("Log Likelihood", fontweight="bold")
    ax.set_xlabel("Iteration", fontweight="bold")
    ax.set_title("Second Model: Training Progress", fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3, linewidth=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"second_model_loss_{filename}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("run predictions")
    # this part is now different

    # for random baseline, calculate the average error
    avg_error = sum(train_df["error"]) / len(train_df["error"])

    mse_baseline = np.mean((test_df["error"] - avg_error) ** 2)

    preds = np.zeros(test_df["user_prompt"].count(), dtype=float)
    i = 0
    for idx in tqdm(test_df["user_prompt"].keys()):
        text = test_df["user_prompt"][idx]
        preds[i] = predict_cont(text, b, vocab_index)
        i += 1

    mse_model = np.mean((test_df["error"] - preds) ** 2)

    # Plot 2: Distribution of Predictions
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.hist(
        preds, bins=100, color="#E63946", edgecolor="black", linewidth=1.2, alpha=0.85
    )
    ax.set_xlabel("Predicted Error", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title(
        "Distribution of Model Predictions per Prompt", fontweight="bold", pad=20
    )
    ax.grid(True, alpha=0.3, axis="y", linewidth=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add text box with MSE results
    textstr = f"MSE Baseline: {mse_baseline:.4f}\nMSE Model: {mse_model:.4f}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax.text(
        0.65,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=20,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()
    plt.savefig(
        f"second_model_prediction_histogram_{filename.split('.')[0]}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"mse_baseline:{mse_baseline}")
    print(f"mse_model:{mse_model}")


if __name__ == "__main__":
    main()

