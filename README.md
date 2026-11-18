# Sentiment of movie reviews: from words to numbers to an LSTM

> **After this tutorial you will be able to** explain how a computer turns a movie review into numbers, what an embedding
> and an LSTM are, train a first model on the IMDB reviews, and judge honestly whether it beats a much simpler method.

[![Open notebook 01 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maryammahmoudi1993/LSTM_emotion_recognition/blob/main/notebooks/01_text_to_numbers_embeddings_lstm.ipynb)

> **Status: in progress.** Notebook 01 is ready. Notebooks 02 and 03 are planned (see the table of contents).

## Who this is for

- **For:** learners who know basic Python and NumPy and have never trained a neural network on text.
- **Not needed:** a GPU, a maths degree, or your own data. Everything runs on a laptop CPU.
- **Time:** about 35 minutes for notebook 01 (about four minutes of training).

## What you will learn (notebook 01)

- How a review becomes a list of word numbers, and why padding is needed.
- What an embedding is (a learned lookup table of coordinates) and what it learned here.
- How an LSTM reads a sentence with a gated memory, and why a plain `SimpleRNN` is unstable.
- Why every neural result needs a simple baseline next to it (here: counting words).

## The pipeline in one picture

```mermaid
flowchart LR
    A["Review<br/>(text)"] --> B["Word numbers<br/>[1, 14, 22, 9, ...]"] --> C["Padding<br/>(200 numbers)"] --> D["Embedding<br/>(200 x 32)"] --> E["LSTM<br/>(reads left to right)"] --> F["Probability<br/>positive"]
```

<p align="center">
  <img src="assets/01_curves.png" alt="Accuracy and loss per epoch of a SimpleRNN and an LSTM on the IMDB reviews, training and validation" width="760"><br>
  <em>The SimpleRNN (blue) is unstable; the LSTM (orange) is good from the first epoch but starts to memorise the training reviews.</em>
</p>

## How to run

**In the browser (no installation):** click the Colab badge above. Colab already has TensorFlow and scikit-learn;
the first cell downloads this repository for you.

**On your computer** (Python 3.13 was tested; any recent Python 3 that TensorFlow supports should work):

```bash
git clone https://github.com/maryammahmoudi1993/LSTM_emotion_recognition.git
cd LSTM_emotion_recognition
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name python3
jupyter lab notebooks/
```

If `jupyter` is not found, run `pip install jupyterlab` first (it is not needed to run the tests).

Check that everything works (runs the smoke tests and every notebook, about four minutes):

```bash
python -m pytest -q
```

The data (IMDB reviews as provided by Keras, about 20 MB) is downloaded automatically on the first run; see [`data/README.md`](data/README.md).

## Contents

| # | Notebook | You will learn | Status |
|---|---|---|---|
| 01 | [From words to numbers: embeddings and a first LSTM](notebooks/01_text_to_numbers_embeddings_lstm.ipynb) | text to numbers, embeddings, LSTM, a simple baseline, honest evaluation | ready |
| 02 | Compare with a small pretrained model on a fixed subset | planned | |
| 03 | Word-level explanations | planned | |

Other folders: [`src/`](src) shared code, [`tests/`](tests) smoke tests and a test that runs every notebook,
[`legacy/`](legacy) the original 2023 files (untouched), [`assets/`](assets) figures and [credits](assets/CREDITS.md).

## Results so far

All numbers come from running notebook 01 (fixed seed 42, CPU, 20,000 training and 5,000 validation reviews, at most 6 epochs,
the weights of the epoch with the best validation accuracy are kept, evaluation on the official 25,000 test reviews, one run each).

| Model | Test accuracy |
|---|---|
| Bag of words + logistic regression (counts words, ignores order) | 0.8572 |
| SimpleRNN (the layer of the first version) | 0.7966 |
| LSTM | 0.8626 |

The LSTM and the word-counting baseline are about half a percentage point apart: from one run we cannot say that the LSTM is better.
Notebook 02 will add a pretrained model on a fixed subset.

## Learning journey

| | **Then (2023, in [`legacy/`](legacy))** | **Now** |
|---|---|---|
| Model | one `SimpleRNN` layer (called "LSTM" in the title and the README) | a real LSTM, next to a `SimpleRNN` and a word-counting baseline |
| Input | 20,000-word vocabulary, 80 words per review, 128 coordinates per word | 10,000-word vocabulary, 200 words per review, 32 coordinates per word (small on purpose) |
| Evaluation | the test reviews were used as validation data in every epoch; no seed | separate training, validation and test sets; seed 42; test set opened once |
| Results | final test accuracy 0.7492 in the saved output, unstable between epochs (0.64 to 0.81) | table above, curves, and a discussion of what the numbers do and do not show |
| Explanation | the README describes the code | notebooks explain every step with pictures and small experiments |

What I would do differently: start with a simple baseline, keep the test set locked, fix the seed, and write down what each step teaches.
See [`legacy/ABOUT.md`](legacy/ABOUT.md) for what I notice in the original notebook today.

## Limitations

- Small models, few epochs and one seed: the numbers show a trend, not the best possible accuracy.
- Reviews are cut to 200 words and rare words are replaced by "unknown".
- The IMDB data set is English movie reviews; the models say nothing about other kinds of text.
- **Licence of the data:** I could not check the terms of use of the data set from the environment where this was written; the data is loaded through Keras and is not part of this repository (see [`data/README.md`](data/README.md)).

## Further reading

- Maas, Daly, Pham, Huang, Ng and Potts (2011), *Learning Word Vectors for Sentiment Analysis*, ACL 2011, <https://aclanthology.org/P11-1015/> (the data set).
- Hochreiter and Schmidhuber (1997), *Long Short-Term Memory*, Neural Computation 9(8), 1735-1780 (the LSTM).
- Keras documentation: [Embedding layer](https://keras.io/api/layers/core_layers/embedding/) and [LSTM layer](https://keras.io/api/layers/recurrent_layers/lstm/).

## License

MIT, see [`LICENSE`](LICENSE). The IMDB data set is not covered by this licence.
