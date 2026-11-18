# Data

This tutorial uses the **IMDB movie review data set** ("Large Movie Review Dataset v1.0") in the form that Keras provides:
50,000 reviews labelled *positive* or *negative* (25,000 for training, 25,000 for testing, half of each class),
where every review is already a list of word numbers.

- Source paper: Maas, Daly, Pham, Huang, Ng and Potts (2011), *Learning Word Vectors for Sentiment Analysis*,
  Proceedings of ACL 2011, pages 142-150, <https://aclanthology.org/P11-1015/>.
- Loaded with `keras.datasets.imdb.load_data()` and `keras.datasets.imdb.get_word_index()`.

You do not download anything by hand: the first time the code runs, Keras stores the files in `~/.keras/datasets/` (about 20 MB).
Nothing is stored inside this repository.

**Licence / terms:** I could not check the terms of use of the data set from the environment in which this tutorial was
written (the page of the authors was not reachable). The data is therefore *not* copied into this repository, only loaded
through Keras. The only review text in the repository is a short excerpt (about 400 characters) of one training review in the saved output of notebook 01, and a few sentences written for teaching.
Please read the terms on the authors' page before using the data for anything else.

## Pretrained vectors (notebook 02)

Notebook 02 downloads the **GloVe** vectors `glove-wiki-gigaword-50` (50 numbers for each of 400,000 words, about 69 MB) from the gensim-data release
<https://github.com/RaRe-Technologies/gensim-data/releases/tag/glove-wiki-gigaword-50>; Keras stores them in `~/.keras/datasets/` after checking a SHA-256 checksum. Nothing is stored in this repository.

- Description in the gensim-data list: pre-trained vectors based on Wikipedia 2014 and Gigaword 5 (5.6 billion tokens, 400,000 words, uncased).
- Licence listed there: Public Domain Dedication and License (<http://opendatacommons.org/licenses/pddl/>). I could not open the original Stanford page to double-check it.
- Paper: Pennington, Socher and Manning (2014), *GloVe: Global Vectors for Word Representation*, EMNLP 2014, <https://aclanthology.org/D14-1162/>.

The ready-made sentiment tool **VADER** (`vaderSentiment` on PyPI, MIT licence, Hutto and Gilbert 2014) is installed from PyPI by `requirements.txt`.

The notebook of the first version of this project (2023) used the same data set; it is kept in [`../legacy`](../legacy).
