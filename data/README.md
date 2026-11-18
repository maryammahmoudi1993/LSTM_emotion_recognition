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
through Keras, and the short example reviews shown in the notebooks are single sentences used for teaching.
Please read the terms on the authors' page before using the data for anything else.

The notebook of the first version of this project (2023) used the same data set; it is kept in [`../legacy`](../legacy).
