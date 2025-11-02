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
        bTx_i = x_i.dot(b)  # prediction
        acc += y_i * (bTx_i) - np.log(1 + np.exp(bTx_i))
    return acc


# collapsing for loop is over 50x faster
def ll_fast(b: ndarray, y: ndarray, x) -> float:
    bTx = x * b
    return float(np.sum(bTx * y - np.log(1 + np.exp(bTx))))


def grad_ll(b: ndarray, y_i: ndarray, x_i: ndarray) -> ndarray:
    return (sigmoid(x_i.dot(b)) - y_i) * x_i


# X is sparse, hence untyped
def logistic_regression(
    X, Y: ndarray, learning_rate: float, num_step: int, loss_capture=1000
) -> tuple[ndarray, ndarray]:
    row_len = X.shape[1]

    b = np.zeros(row_len)
    lls = np.zeros((num_step // loss_capture) + 1)
    for i in tqdm(range(num_step)):
        if i % loss_capture == 0:
            lls[(i // loss_capture)] = ll_fast(b, Y, X)
        rand_index = i % len(Y)
        row = X[rand_index]
        label = Y[rand_index]
        b = b - learning_rate * grad_ll(b, label, row)
    lls[-1] = ll_fast(b, Y, X)

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
    df: pd.DataFrame = pd.read_csv("generated_with_labels.csv")
    train_df, test_df = train_test_split(df, test_size=0.2)

    print(f"train size:{len(train_df)}, test size:{len(test_df)}")
    print(f"train data: {train_df} test data: {test_df}")

    print("before vocab generation")
    vocabulary, vocab_index = derive_vocab(train_df)

    print("before df split")
    docs = train_df["generation"]
    labels = train_df["label"]

    print("before label map generation")

    label_ids = np.array(train_df["label"]).astype(float)
    print(label_ids)

    # next step: create a sparse D x V matrix
    # note a bias term will be eventually needed

    print("before sparse mat population")

    mat_csr = sparse_dv_mat(train_df, vocab_index).tocsr()
    # coo is probably the correct way to do it

    print("before logistic regression")

    epochs = 2

    b, lls = logistic_regression(
        mat_csr, label_ids, 1e-4, epochs * len(label_ids), loss_capture=10
    )
    sns.lineplot(x=range(len(lls)), y=lls)
    plt.savefig("np_best.png")

    # print("load test data")
    # test_df = pd.read_csv("generated_with_labels.csv")

    print("run predictions")
    # res = np.zeros(test_df["generation"].count(), dtype=bool)
    preds = np.zeros(test_df["generation"].count(), dtype=bool)
    correct = np.array(test_df["label"], dtype=bool)

    ct = 0
    for i in tqdm(test_df["generation"].keys()):
        text = test_df["generation"][i]
        preds[ct] = predict(text, b, vocab_index)
        true_label = test_df["label"][i]
        ct += 1

    print(f"accuracy:{accuracy(preds, correct)}")
    print(f"f1_score:{f1_score(preds, correct)}")

    # read in original data, and then use this to create data for a secondary classification task
    gen_df = pd.read_csv("generated.csv")

    # for each prompt, error is the difference between the human and AI generated text
    # is there a double dipping issue?

    # for now, ignore it...
    # ai is 1, human is 0
    errors = []
    for i in range(len(gen_df["ai_output"])):
        ai_gen = str(gen_df["ai_output"][i])
        human_gen = str(gen_df["human_output"][i])
        ai_pred = predict_cont(ai_gen, b, vocab_index)
        human_pred = predict_cont(human_gen, b, vocab_index)
        # print(ai_pred, human_pred)

        # multiple ways to prhase this error
        error = abs(1 - ai_pred) + abs(human_pred - 0)
        assert error == (1 - ai_pred) + human_pred
        errors.append(error)

        # error = abs(ai_pred - human_pred)
        # gen_df["error"][i] = error

    gen_df["error"] = np.array(errors)
    print(errors)
    gen_df.to_csv("generated_post_classification.csv", index=False)


if __name__ == "__main__":
    main()
