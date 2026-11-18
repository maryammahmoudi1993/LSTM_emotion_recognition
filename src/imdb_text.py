"""Helpers shared by the notebooks: data, text-to-numbers, models and figures."""

import os
import re
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")     # keep TensorFlow start-up messages quiet
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")    # avoid tiny run-to-run rounding differences

import keras
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from keras.utils import pad_sequences

SEED = 42
VOCAB = 10000          # we keep the 10,000 most frequent words; every other word becomes "unknown"
MAXLEN = 200           # every review is cut or padded to 200 words
RESERVED = {0: "<pad>", 1: "<start>", 2: "<unknown>"}   # ids 0, 1, 2 are special; real words start at 3
ASSETS = Path(__file__).resolve().parents[1] / "assets"


def set_seed(seed=SEED):
    """Fix every source of randomness we use, so reruns give the same numbers on a CPU."""
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def save_figure(fig, name):
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / name, dpi=110, bbox_inches="tight")


# ---------------------------------------------------------------- data and vocabulary
def load_reviews(n_train=20000):
    """Return ((train_x, train_y), (val_x, val_y), (test_x, test_y)).

    The x values are Python lists of word numbers (reviews have different lengths).
    The 25,000 official training reviews are shuffled once with a fixed seed: the first ``n_train`` are used for
    training and the rest for validation. The 25,000 official test reviews are never used for any decision.
    """
    (x_all, y_all), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=VOCAB)
    order = np.random.default_rng(SEED).permutation(len(x_all))
    x_all = [x_all[i] for i in order]
    y_all = np.asarray(y_all)[order]
    return ((x_all[:n_train], y_all[:n_train]),
            (x_all[n_train:], y_all[n_train:]),
            (list(x_test), np.asarray(y_test)))


def word_to_id():
    """Dictionary word -> number, with the three reserved numbers already taken into account."""
    raw = keras.datasets.imdb.get_word_index()
    return {word: number + 3 for word, number in raw.items() if number + 3 < VOCAB}


def id_to_word(mapping=None):
    """Dictionary number -> word (including the reserved symbols)."""
    mapping = mapping or word_to_id()
    inverse = {number: word for word, number in mapping.items()}
    inverse.update(RESERVED)
    return inverse


def decode(sequence, inverse):
    """Turn a list of word numbers back into text (skipping the start symbol)."""
    return " ".join(inverse.get(int(n), "<unknown>") for n in sequence if int(n) != 1)


def encode(text, mapping):
    """Turn a sentence into word numbers: lower case, letters and apostrophes only, unknown words -> 2."""
    words = re.findall(r"[a-z']+", text.lower())
    return [1] + [mapping.get(word, 2) for word in words]


def pad(sequences, maxlen=MAXLEN):
    """Make all reviews exactly ``maxlen`` long: short ones are padded with 0 at the *front*,
    long ones lose their *first* words, so the real text always ends at the last position."""
    return pad_sequences(sequences, maxlen=maxlen, padding="pre", truncating="pre")


def bag_of_words(sequences):
    """Count how often each word occurs in each review (a sparse matrix, order of words is ignored)."""
    rows, cols = [], []
    for i, review in enumerate(sequences):
        rows.extend([i] * len(review))
        cols.extend(review)
    data = np.ones(len(rows), dtype="float32")
    return sp.csr_matrix((data, (rows, cols)), shape=(len(sequences), VOCAB))


# ---------------------------------------------------------------- models
def build_model(kind="lstm", embedding_dim=32, units=32):
    """Embedding -> recurrent layer -> one output between 0 (negative) and 1 (positive)."""
    recurrent = {"lstm": keras.layers.LSTM, "simple_rnn": keras.layers.SimpleRNN}[kind]
    return keras.Sequential([
        keras.layers.Input((MAXLEN,)),
        keras.layers.Embedding(VOCAB, embedding_dim),
        recurrent(units),
        keras.layers.Dense(1, activation="sigmoid"),
    ], name=kind)


def compile_model(model):
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train(kind, x_train, y_train, x_val, y_val, epochs=6, batch_size=64):
    """Train one model and return (model, history). The weights of the epoch with the best
    *validation* accuracy are restored at the end (the test set is not involved)."""
    set_seed()
    model = compile_model(build_model(kind))
    stop = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=epochs,
                                         restore_best_weights=True)
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(x_val, y_val), callbacks=[stop], verbose=0)
    return model, history.history


def cosine_neighbours(matrix, mapping, inverse, word, k=6):
    """The ``k`` words whose embedding vectors point in the most similar direction as ``word``."""
    vectors = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    similarity = vectors @ vectors[mapping[word]]
    best = [i for i in np.argsort(-similarity) if i > 2 and i != mapping[word]][:k]
    return [(inverse[i], float(similarity[i])) for i in best]


# ---------------------------------------------------------------- figures
def plot_lengths(lengths, labels):
    """Histogram of review lengths and the class balance."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [2.2, 1]})
    a.hist(lengths, bins=60, color="tab:blue", alpha=.8)
    a.axvline(MAXLEN, color="tab:red", ls="--", label=f"we keep {MAXLEN} words")
    a.set_xlabel("words per review"); a.set_ylabel("number of reviews")
    a.set_title("how long are the reviews?"); a.legend()
    counts = np.bincount(labels)
    b.bar(["negative", "positive"], counts, color=["tab:red", "tab:green"])
    b.set_title("class balance (training)")
    b.set_ylim(0, counts.max() * 1.12)
    for x, c in enumerate(counts):
        b.text(x, c, str(c), ha="center", va="bottom")
    return fig


def plot_pipeline():
    """The path from a sentence to a probability, drawn with boxes."""
    labels = ["review\n(text)", "word numbers\n[1, 14, 22, 9, ...]", "padding\n(200 numbers)",
              "embedding\n(200 x 32 numbers)", "LSTM\n(reads left to right)", "probability\npositive"]
    fig, ax = plt.subplots(figsize=(13, 2.2))
    ax.axis("off")
    for k, text in enumerate(labels):
        ax.text(k, 0, text, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", fc="#e8f0fe", ec="#4a6fa5"))
        if k:
            ax.annotate("", xy=(k - 0.32, 0), xytext=(k - 0.68, 0),
                        arrowprops=dict(arrowstyle="->", color="#4a6fa5"))
    ax.set_xlim(-0.6, len(labels) - 0.4); ax.set_ylim(-0.6, 0.6)
    return fig


def plot_history(histories, labels):
    """Training and validation accuracy and loss for several models (dashed = training)."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for k, (h, label) in enumerate(zip(histories, labels)):
        color = f"C{k}"
        epochs = range(1, len(h["accuracy"]) + 1)
        a.plot(epochs, h["accuracy"], "--", color=color, alpha=.8, label=f"{label}: training")
        a.plot(epochs, h["val_accuracy"], "-", color=color, marker="o", label=f"{label}: validation")
        b.plot(epochs, h["loss"], "--", color=color, alpha=.8, label=f"{label}: training")
        b.plot(epochs, h["val_loss"], "-", color=color, marker="o", label=f"{label}: validation")
    a.set_title("accuracy (higher is better)"); b.set_title("loss (lower is better)")
    for ax in (a, b):
        ax.set_xlabel("epoch"); ax.grid(alpha=.3)
    a.legend(fontsize=7)
    return fig


def plot_embedding_map(matrix, mapping, positive, negative):
    """Project the embedding vectors of some words to two dimensions (PCA) and plot them."""
    words = positive + negative
    vectors = np.array([matrix[mapping[w]] for w in words])
    vectors = vectors - vectors.mean(axis=0)
    _, _, vt = np.linalg.svd(vectors, full_matrices=False)
    points = vectors @ vt[:2].T
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for k, word in enumerate(words):
        color = "tab:green" if k < len(positive) else "tab:red"
        ax.scatter(*points[k], color=color)
        ax.annotate(word, points[k], fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_title("embedding vectors of sentiment words (2-D projection)\ngreen = positive words, red = negative words")
    ax.set_xlabel("first direction"); ax.set_ylabel("second direction")
    return fig
