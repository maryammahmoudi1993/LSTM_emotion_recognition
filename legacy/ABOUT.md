# Legacy files

These are my original files from the first version of this project (January and February 2023), kept
unchanged: `imbd_emotions.ipynb` (the notebook, with its saved output) and the first `README.md`.

What I notice today, from reading the notebook and its saved output:

- The repository is called "LSTM", the README says "LSTM (RNN)", but the model is a **`SimpleRNN`** layer, not an LSTM.
- The test set is used as `validation_data` in every epoch, so the curves are drawn on the very reviews that should stay
  locked until the end; the test set is no longer a fresh exam.
- The saved output shows an unstable training: the training accuracy climbs from 0.57 to 0.93 over 15 epochs,
  while the accuracy on the test reviews jumps between 0.64 and 0.81 and ends at 0.7492 (final evaluation in the notebook).
  A big gap between training and test accuracy is the typical sign of memorising.
- No random seed is set, so a rerun gives different numbers; there is no baseline to compare with, no explanation of the
  numbers, and the README describes the code but does not show results.

The new notebooks in `../notebooks/` rebuild the project step by step: text to numbers, embeddings, a real LSTM, an honest
comparison with a simple baseline, and (later) a comparison with a small pretrained model and word-level explanations.
