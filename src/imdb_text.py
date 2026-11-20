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
def build_model(kind="lstm", embedding_dim=32, units=32, embedding_matrix=None, trainable=True):
    """Embedding -> recurrent layer -> one output between 0 (negative) and 1 (positive).

    If ``embedding_matrix`` (VOCAB x dimension) is given, the embedding table starts from these
    pretrained vectors; with ``trainable=False`` it stays frozen during training.
    """
    recurrent = {"lstm": keras.layers.LSTM, "simple_rnn": keras.layers.SimpleRNN}[kind]
    if embedding_matrix is None:
        embedding = keras.layers.Embedding(VOCAB, embedding_dim)
    else:
        embedding = keras.layers.Embedding(
            VOCAB, embedding_matrix.shape[1],
            embeddings_initializer=keras.initializers.Constant(embedding_matrix), trainable=trainable)
    return keras.Sequential([
        keras.layers.Input((MAXLEN,)),
        embedding,
        recurrent(units),
        keras.layers.Dense(1, activation="sigmoid"),
    ], name=kind)


def compile_model(model):
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train(kind, x_train, y_train, x_val, y_val, epochs=6, batch_size=64, seed=SEED, **model_options):
    """Train one model and return (model, history). The weights of the epoch with the best
    *validation* accuracy are restored at the end (the test set is not involved)."""
    set_seed(seed)
    model = compile_model(build_model(kind, **model_options))
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


# ---------------------------------------------------------------- pretrained pieces
GLOVE_URL = ("https://github.com/RaRe-Technologies/gensim-data/releases/download/"
             "glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz")
GLOVE_SHA256 = "5c55f98957aa9fed8d2ac5fb1dcff57af3b23c5a3ee7af3f7945f8d49198eb24"


def download_glove():
    """Download the GloVe vectors (50 numbers for each of 400,000 words, about 69 MB, stored by Keras
    in ~/.keras/datasets, not in this repository) and check their checksum."""
    return keras.utils.get_file("glove-wiki-gigaword-50.gz", GLOVE_URL, file_hash=GLOVE_SHA256,
                                hash_algorithm="sha256")


def glove_matrix(path, mapping, dimension=50):
    """Build the embedding table for our 10,000 words from the GloVe file.

    Returns (matrix, found): words that GloVe does not know (and the three reserved numbers) get
    small random numbers; ``found`` counts the words that did get a pretrained vector.
    """
    import gzip
    rng = np.random.default_rng(SEED)
    matrix = rng.normal(0.0, 0.05, size=(VOCAB, dimension)).astype("float32")
    matrix[0] = 0.0
    found = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        next(handle)                                   # header line: "400000 50"
        for line in handle:
            word, _, rest = line.partition(" ")
            index = mapping.get(word)
            if index is not None:
                matrix[index] = np.asarray(rest.split(), dtype="float32")
                found += 1
    return matrix, found


def full_text_reviews():
    """The 25,000 test reviews as text rebuilt from the *full* word list (no 10,000-word limit).

    The original reviews are only stored as word numbers: capital letters and punctuation are gone.
    """
    raw = keras.datasets.imdb.get_word_index()
    inverse = {number + 3: word for word, number in raw.items()}
    inverse.update(RESERVED)
    (_, _), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=None)
    texts = [" ".join(w for w in (inverse.get(int(n), "") for n in review if int(n) > 2) if w != "br")
             for review in x_test]                     # "br" is what is left of the line breaks <br />
    return texts, np.asarray(y_test)


def vader_predictions(texts):
    """VADER (Hutto and Gilbert, 2014): a hand-made sentiment dictionary plus a few rules, no training.
    Returns 1 (positive) when the overall "compound" score is above 0, otherwise 0."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    return np.array([int(analyzer.polarity_scores(text)["compound"] > 0) for text in texts])


# ---------------------------------------------------------------- explanations (word level)
def tokenize(text):
    """The words of a text, exactly as ``encode`` sees them."""
    return re.findall(r"[a-z']+", text.lower())


def ids_of(words, mapping):
    """Word numbers of a list of words (with the start symbol in front)."""
    return [1] + [mapping.get(word, 2) for word in words]


def probability_positive(model, words_list, mapping):
    """Probability of "positive" for several reviews, each given as a list of words."""
    batch = pad([ids_of(words, mapping) for words in words_list])
    return model.predict(batch, batch_size=256, verbose=0)[:, 0]


def integrated_gradients(model, words, mapping, steps=64):
    """Integrated gradients (Sundararajan, Taly and Yan, 2017) for one review.

    The review is walked from an "empty" input (all embedding vectors zero) to the real input in ``steps`` small steps;
    the gradients of p(positive) with respect to the embedding vectors are averaged along the way and multiplied by the
    total change. Returns (score per word position, completeness gap). The scores add up to
    p(positive) of the review minus p(positive) of the empty input; the gap shows how well that holds.
    """
    x = pad([ids_of(words, mapping)])
    embedding, rest = model.layers[0], model.layers[1:]
    e = embedding(x)                                           # (1, 200, dimension)
    baseline = tf.zeros_like(e)
    alphas = tf.linspace(0.0, 1.0, steps + 1)[1:]
    path = baseline + alphas[:, None, None] * (e - baseline)   # (steps, 200, dimension)
    with tf.GradientTape() as tape:
        tape.watch(path)
        h = path
        for layer in rest:
            h = layer(h)
        output = h[:, 0]                                       # p(positive) for every step of the path
    gradients = tape.gradient(output, path)
    scores = tf.reduce_sum((e[0] - baseline[0]) * tf.reduce_mean(gradients, axis=0), axis=-1).numpy()
    h0 = baseline
    for layer in rest:
        h0 = layer(h0)
    total = float(model(x)[0, 0]) - float(h0[0, 0])
    n_words = len(words)
    return scores[-n_words:], float(scores.sum()) - total     # last positions hold the real words


def by_word(words, scores):
    """Add up the scores of a word that occurs several times: {word: total score}."""
    totals = {}
    for word, score in zip(words, scores):
        totals[word] = totals.get(word, 0.0) + float(score)
    return totals


def occlusion_scores(model, words, mapping):
    """For every distinct word: how much p(positive) drops when all its occurrences are deleted."""
    distinct = list(dict.fromkeys(words))
    variants = [[w for w in words if w != word] for word in distinct]
    base = probability_positive(model, [words], mapping)[0]
    after = probability_positive(model, variants, mapping)
    return {word: float(base - p) for word, p in zip(distinct, after)}


def lime_scores(model, words, mapping, seed=SEED, num_samples=1000):
    """LIME (Ribeiro, Singh and Guestrin, 2016) for one review: {word: weight toward "positive"}."""
    from lime.lime_text import LimeTextExplainer

    def predict(texts):
        p = probability_positive(model, [tokenize(t) for t in texts], mapping)
        return np.column_stack([1 - p, p])

    explainer = LimeTextExplainer(class_names=["negative", "positive"], random_state=seed, bow=True)
    explanation = explainer.explain_instance(" ".join(words), predict, labels=(1,), num_features=len(set(words)),
                                             num_samples=num_samples)
    return dict(explanation.as_list(label=1))


def delete_words(words, to_delete):
    """The review without the given words (all occurrences)."""
    drop = set(to_delete)
    return [w for w in words if w not in drop]


def draw_words(ax, words, scores, per_line=9, limit=None):
    """Write a review with every word coloured by its score (green = toward positive, red = toward negative)."""
    limit = limit or max(1e-9, max(abs(v) for v in scores))
    ax.axis("off")
    lines = [list(zip(words[i:i + per_line], scores[i:i + per_line])) for i in range(0, len(words), per_line)]
    for row, line in enumerate(lines):
        x = 0.0
        for word, score in line:
            strength = min(1.0, abs(score) / limit)
            color = (0.2, 0.65, 0.2, 0.15 + 0.85 * strength) if score > 0 else (0.85, 0.15, 0.15, 0.15 + 0.85 * strength)
            ax.text(x, -row, word, fontsize=9, va="center", bbox=dict(boxstyle="round,pad=0.2", fc=color, ec="none"))
            x += 0.012 * (len(word) + 2) + 0.01
    ax.set_xlim(0, 1.15); ax.set_ylim(-len(lines), 1)


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


def plot_results(names, means, lows, highs, colors=None):
    """Horizontal bars of test accuracy; the thin line spans the smallest to the largest value over seeds."""
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    positions = np.arange(len(names))[::-1]
    ax.barh(positions, means, color=colors or "tab:blue", alpha=.85)
    ax.hlines(positions, lows, highs, color="black", lw=1.5)
    for y, m in zip(positions, means):
        ax.text(m + 0.004, y, f"{m:.3f}", va="center", fontsize=9)
    ax.set_yticks(positions); ax.set_yticklabels(names)
    ax.set_xlim(0.5, 0.95); ax.set_xlabel("test accuracy on 25,000 reviews (higher is better)")
    ax.grid(axis="x", alpha=.3)
    return fig


def plot_top_words(words, scores, title, k=12):
    """Horizontal bars of the ``k`` most positive and the ``k`` most negative words."""
    order = np.argsort(scores)
    pick = list(order[:k]) + list(order[-k:])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.barh(range(len(pick)), [scores[i] for i in pick],
            color=["tab:red" if scores[i] < 0 else "tab:green" for i in pick])
    ax.set_yticks(range(len(pick))); ax.set_yticklabels([words[i] for i in pick])
    ax.axvline(0, color="black", lw=.8); ax.set_title(title)
    ax.set_xlabel("weight (left: toward negative, right: toward positive)")
    return fig


def plot_deletion(names, means, medians):
    """Mean (bar) and median (dot) drop of the predicted-class probability after deleting the top-3 words."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    positions = np.arange(len(names))[::-1]
    ax.barh(positions, means, color="tab:blue", alpha=.85)
    ax.plot(medians, positions, "ko", label="median")
    for y, m in zip(positions, means):
        ax.text(m + 0.006, y, f"{m:.3f}", va="center", fontsize=9)
    ax.set_yticks(positions); ax.set_yticklabels(names)
    ax.set_xlabel("drop of the predicted class's probability after deleting 3 words (larger = the words mattered more)")
    ax.legend(); ax.grid(axis="x", alpha=.3)
    return fig
