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


def test_pretrained_embedding_matrix_is_used(tmp_path):
    import gzip
    mapping = {"good": 5, "bad": 6}
    path = tmp_path / "tiny.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("2 3\ngood 1 2 3\nbad -1 -2 -3\n")
    matrix, found = it.glove_matrix(path, mapping, dimension=3)
    assert found == 2 and matrix.shape == (it.VOCAB, 3)
    assert list(matrix[5]) == [1, 2, 3] and list(matrix[6]) == [-1, -2, -3]
    model = it.build_model("lstm", embedding_matrix=matrix, trainable=False)
    assert np.allclose(model.layers[0].get_weights()[0][5], [1, 2, 3])
    assert not model.layers[0].trainable and it.build_model("lstm").layers[0].trainable


def test_vader_reads_obvious_sentences():
    predictions = it.vader_predictions(["a wonderful and brilliant movie", "a terrible and boring movie"])
    assert list(predictions) == [1, 0]


def test_explanation_helpers_on_a_tiny_model():
    mapping = it.word_to_id()
    it.set_seed()
    model = it.compile_model(it.build_model("lstm", units=8))
    words = it.tokenize("a great movie but a boring ending")
    scores, gap = it.integrated_gradients(model, words, mapping, steps=16)
    assert scores.shape == (len(words),) and abs(gap) < 0.05             # scores add up to the change of the output
    occlusion = it.occlusion_scores(model, words, mapping)
    assert set(occlusion) == set(words) and it.by_word(["a", "a", "b"], [1, 2, 3]) == {"a": 3.0, "b": 3.0}
    assert it.delete_words(words, ["a"]) == ["great", "movie", "but", "boring", "ending"]
    lime = it.lime_scores(model, words, mapping, num_samples=60)
    assert set(lime) <= set(words) and len(lime) > 0
