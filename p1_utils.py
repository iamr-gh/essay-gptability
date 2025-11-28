from collections import Counter
import re
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
# this file contains all the shared utils between the torch and numpy implems


def tokenize(data: str) -> list[str]:
    return data.split(" ")


def canonicalize(word: str) -> str:
    return re.sub(r"\.|,|'|\"|\(|\)", "", word.lower())


def better_tokenize(data: str) -> list[str]:
    original = tokenize(data)

    # further split by newline
    separated = []
    for tok in original:
        separated.extend(tok.split("\n"))

    out = [canonicalize(word) for word in separated]
    return out


def derive_vocab(
    df: pd.DataFrame, key="generation"
) -> tuple[list[str], dict[str, int]]:
    # dictionary require minimum of 250 word frequency for inclusion
    all_words = Counter()
    for row in tqdm(df[key]):
        words = better_tokenize(row)
        all_words.update(words)

    vocabulary = []
    min_word_freq = 250
    for word, freq in all_words.items():
        if freq >= min_word_freq:
            vocabulary.append(word)

    # note there are some unprintable tokens in here
    vocab_index = {}
    for i, word in enumerate(vocabulary):
        vocab_index[word] = i

    return vocabulary, vocab_index


def sparse_dv_mat(docs_df: pd.DataFrame, vocab_index: dict[str, int], key="generation"):
    docs = docs_df[key]
    rows = []
    cols = []
    vals = []
    for row, doc in tqdm(enumerate(docs)):
        words = better_tokenize(doc)
        counts = Counter(words)
        for word, count in counts.items():
            if word in vocab_index:
                rows.append(row)
                col = vocab_index[word]
                cols.append(col)
                vals.append(count)
        # bias
        rows.append(row)
        col = len(vocab_index.keys())
        cols.append(col)
        vals.append(1)

    mat_coo = sparse.coo_matrix(
        (vals, (rows, cols)),
        shape=(len(docs), len(vocab_index.keys()) + 1),
        dtype=float,
    )
    return mat_coo


def f1_score(pred, actual) -> float:
    # two arrays of 1/0
    tp_arr = pred & actual
    fp_arr = pred & ~actual
    fn_arr = ~pred & actual

    tp = tp_arr.sum()
    fp = fp_arr.sum()
    fn = fn_arr.sum()

    return (2 * tp) / (2 * tp + fp + fn)


# tp + tn
def accuracy(pred, actual) -> float:
    # two arrays of bool
    tp_arr = pred & actual
    fp_arr = pred & ~actual
    fn_arr = ~pred & actual
    tn_arr = ~pred & ~actual

    tp = tp_arr.sum()
    fp = fp_arr.sum()
    fn = fn_arr.sum()
    tn = tn_arr.sum()

    return (tp + tn) / len(pred)
