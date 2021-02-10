import numpy as np
import tensorflow as tf
from preprocess import *

class RNN_Part1(tf.keras.Model):
  def __init__(self, vocab):
    """
        The RNN_Part1 class predicts the next words in a sequence.
        Feel free to initialize any variables that you find necessary in the constructor.

        :param vocab_size: The number of unique words in the data
        """

    super(RNN_Part1, self).__init__()

    self.vocab_size = len(vocab)
    self.vocab = vocab
    output_dim_embed = 128 
    dimension_gru = output_dim_embed
    dense_output_size = self.vocab_size # I think it has to be this

    self.embedding_layer = tf.keras.layers.Embedding(self.vocab_size, output_dim_embed, embeddings_initializer=tf.keras.initializers.RandomNormal)
    self.gru_layer = tf.keras.layers.GRU(dimension_gru, return_sequences=True)
    self.dense_layer = tf.keras.layers.Dense(dense_output_size, activation='softmax')

    # TODO: initialize tf.keras.layers!
    # - tf.keras.layers.Embedding for embedding layers: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
    # - tf.keras.layers.Dense for feed forward layers: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
    # - and use tf.keras.layers.GRU or tf.keras.layers.LSTM for your RNN


  def call(self, inputs):
    """
        - You must use an embedding layer as the first layer of your network 
        - You must use an LSTM or GRU as the next layer.

        :param inputs: word ids of shape (batch_size, window_size)
        :return: the batch element probabilities as a tensor
        """
    # TODO: implement the forward pass calls on your tf.keras.layers!

    # Embedding layer
    embedding = self.embedding_layer(inputs)

    # LSTM or GRU layer
    gru_result = self.gru_layer(embedding)

    # Densely connected layer
    results = self.dense_layer(gru_result)

    return results

  def loss(self, probs, labels):
    """
        Calculates average cross entropy sequence to sequence loss of the prediction

        :param probs: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """

    # We recommend using tf.keras.losses.sparse_categorical_crossentropy
    # https://www.tensorflow.org/api_docs/python/tf/keras/losses/sparse_categorical_crossentropy

    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)

    # TODO: implement the loss function with mask as described in the writeup
    mask = tf.less(labels, self.vocab[FIRST_SPECIAL]) | tf.greater(labels, self.vocab[LAST_SPECIAL])
    loss = tf.boolean_mask(loss, mask)
    average_loss = tf.reduce_mean(loss)

    return average_loss

class RNN_Part2(tf.keras.Model):
  def __init__(self, french_vocab, english_vocab):

    super(RNN_Part2, self).__init__()

    french_vocab_size = len(french_vocab)
    english_vocab_size = len(english_vocab)

    self.french_vocab = french_vocab
    self.english_vocab = english_vocab

    self.french_vocab_size = len(french_vocab)
    self.english_vocab_size = len(english_vocab)
    output_dim_embed = 800
    dimension_gru = output_dim_embed
    endense_output_size = self.english_vocab_size

    # French layers (encoder)
    self.french_embedding_layer = tf.keras.layers.Embedding(self.french_vocab_size, output_dim_embed)
    self.french_dense_layer_1 = tf.keras.layers.Dense(output_dim_embed, activation='relu')
    self.french_gru_layer_1 = tf.keras.layers.GRU(dimension_gru, return_sequences=True, return_state=True, recurrent_activation='softmax')
    self.french_gru_layer_2 = tf.keras.layers.GRU(dimension_gru, return_sequences=True, return_state=True, recurrent_activation='softmax')
    # English layers (decoder)
    self.english_embedding_layer = tf.keras.layers.Embedding(self.english_vocab_size, output_dim_embed)
    self.english_gru_layer_1 = tf.keras.layers.GRU(dimension_gru, return_sequences=True, return_state=True, recurrent_activation='softmax')
    self.english_gru_layer_2 = tf.keras.layers.GRU(dimension_gru, return_sequences=True, return_state=True, recurrent_activation='softmax')
    self.english_dense_layer = tf.keras.layers.Dense(endense_output_size, activation='softmax')
    self.english_dense_layer_1 = tf.keras.layers.Dense(endense_output_size, activation='softmax')

  def call(self, encoder_input, decoder_input):
    """
    :param encoder_input: batched ids corresponding to french sentences
    :param decoder_input: batched ids corresponding to english sentences
    :return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
    """
    # TODO: implement the forward pass calls on your tf.keras.layers!
    # Note 1: in the diagram there are two inputs to the decoder
    #  (the decoder_input and the hidden state of the encoder)
    #  Be careful because we don't actually need the predictive output
    #   of the encoder -- only its hidden state
    # Note 2: If you use an LSTM, the hidden_state will be the last two
    #   outputs of calling the rnn. If you use a GRU, it will just be the
    #   second output.

    # Embedding
    french_embedding = self.french_embedding_layer(encoder_input)
    french_embedding = self.french_dense_layer_1(french_embedding)
    english_embedding = self.english_embedding_layer(decoder_input)

    # Encoder
    french_gru_result_1, hidden_state_1 = self.french_gru_layer_1(french_embedding)
    french_gru_result_2, hidden_state_2 = self.french_gru_layer_2(french_gru_result_1, initial_state=hidden_state_1)

    # Decoding
    english_gru_result_1, hidden_state_3 = self.english_gru_layer_1(english_embedding, initial_state=hidden_state_2)
    english_gru_result_2, hidden_state_4 = self.english_gru_layer_2(english_gru_result_1, hidden_state_3)
    decoder_result = self.english_dense_layer(english_gru_result_2)
    # decoder_result = self.english_dense_layer(english_gru_result_2)

    return decoder_result

  def loss_function(self, probs, labels):
    """
    Calculates the model cross-entropy loss after one forward pass.

    :param probs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
    :param labels:  integer tensor, word prediction labels [batch_size x window_size]
    :return: the loss of the model as a tensor
    """


    # When computing loss, we need to compare the output probs and labels with a shift
    #  of 1 to ensure a proper alignment. This is because we generated the output by passing
    #  in a *START* token and the encoded French state.
    #
    # - The labels should have the first token removed:
    #   [*START* COSC440 is the best class. *STOP*] --> [COSC440 is the best class. *STOP*]
    # - The logits should have the last token in the window removed:
    #   [COSC440 is the best class. *STOP* *PAD*] --> [COSC440 is the best class. *STOP*]

      # TODO: implement the loss function with mask as described in the writeup

    mask_value = 43

    labels = labels[1:]
    probs = probs[:-1]

    ##################### I THINK THE MASK IS WRONG??? #####################################
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
    mask = tf.greater(mask_value, labels)
    mask = tf.math.logical_not(mask)
    # print("LABELS:", labels[0])
    # print("MASK:", mask[0])
    #mask = tf.less(labels, self.english_vocab[FIRST_SPECIAL]) | tf.greater(labels, self.english_vocab[LAST_SPECIAL])
    loss = tf.boolean_mask(loss, mask)
    average_loss = tf.reduce_mean(loss)

    return average_loss
