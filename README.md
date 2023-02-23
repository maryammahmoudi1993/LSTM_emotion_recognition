# LSTM Emotion Recognition in IMDB Dataset
This project uses LSTM (RNN) networks in order to find positive or negative comments. IMDB dataset used for training.

This code is a simple implementation of emotion recognition using a SimpleRNN neural network trained on the IMDB dataset. The code is written in Python and uses the Keras library.

# Dataset
The code uses the IMDB dataset, which is a set of 50,000 highly polarized movie reviews. The dataset is divided into 25,000 reviews for training and 25,000 reviews for testing. The reviews are labeled as either positive or negative. The dataset is loaded using the imdb.load_data() function, which loads only the top max_features most frequently occurring words in the dataset. The maximum length of a sequence is maxlen, which is set to 80.

# Preprocessing
The pad_sequences() function from the Keras library is used to normalize all sequences. This function adds zero padding to the beginning of each sequence to make them all the same length. The resulting sequences are used as inputs to the neural network.

# Model Architecture
The neural network consists of three layers:

Embedding layer: This layer converts each word in the input sequence to a vector of real numbers. The size of the output vectors is set to 128.
SimpleRNN layer: This layer is a type of recurrent neural network that processes the sequence of input vectors generated by the embedding layer. The size of the layer is also set to 128. Dropout and recurrent dropout are set to 0.2 to prevent overfitting.
Dense layer: This layer consists of a single node with a sigmoid activation function. This node outputs the predicted probability that the input sequence belongs to the positive class.
The neural network is compiled using the binary cross-entropy loss function and the Adam optimizer. The accuracy metric is also computed during training.

# Training
The neural network is trained using the fit() function of the Keras library. The training data is passed as inputs, along with the batch size and number of epochs. The validation data is also passed to this function, allowing the model to be evaluated on the test set after each epoch.

# Results
The performance of the model is visualized using the show_results() function. This function plots the training and test accuracy and loss over each epoch. The final accuracy and loss on the test set are also printed.
