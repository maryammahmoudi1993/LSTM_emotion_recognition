"""Quick checks that the helpers work. They do not train anything for long."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import imdb_text as it  # noqa: E402


def test_data_and_vocabulary():
    (x, y), (xv, yv), (xt, yt) = it.load_reviews(n_train=2000)
    assert len(x) == 2000 and len(xv) == 23000 and len(xt) == 25000
    assert set(np.unique(y)) == {0, 1}
    assert max(max(review) for review in x[:200]) < it.VOCAB
    mapping = it.word_to_id()
    inverse = it.id_to_word(mapping)
    assert inverse[mapping["great"]] == "great" and inverse[0] == "<pad>"


def test_encode_decode_and_pad():
    mapping = it.word_to_id()
    inverse = it.id_to_word(mapping)
    ids = it.encode("This movie was GREAT, zzzzqqq!", mapping)
    assert ids[0] == 1 and ids[-1] == 2                      # start symbol, unknown word
    assert it.decode(ids, inverse) == "this movie was great <unknown>"
    padded = it.pad([[5, 6, 7], list(range(3, 400))])
    assert padded.shape == (2, it.MAXLEN)
    assert list(padded[0][-3:]) == [5, 6, 7] and padded[0][0] == 0   # padded at the front
    assert padded[1][-1] == 399                                       # long review keeps its end


def test_bag_of_words_counts_words():
    bow = it.bag_of_words([[3, 3, 4], [5]])
    assert bow.shape == (2, it.VOCAB) and bow[0, 3] == 2 and bow[0, 4] == 1 and bow[1, 5] == 1


def test_models_output_shape():
    batch = np.zeros((2, it.MAXLEN), dtype="int32")
    for kind in ("lstm", "simple_rnn"):
        assert it.build_model(kind)(batch).shape == (2, 1)


def test_training_is_reproducible():
    (x, y), (xv, yv), _ = it.load_reviews(n_train=200)
    x, xv = it.pad(x), it.pad(xv[:100])
    losses = []
    for _ in range(2):
        _, history = it.train("lstm", x, y, xv, yv[:100], epochs=1)
        losses.append(history["loss"][0])
    assert losses[0] == losses[1]
