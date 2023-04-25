

import tensorflow as tf

class biLSTM(tf.keras.Model):
    def __init__(self, vocab_size, embeddings=None, embedding_dim=100, hidden_dim=256, output_dim=3, pad_idx=1, unk_idx=0):
        """
        pad_idx: the index of the padding token <pad> in the vocabulary
        num_layers: number of biLSTMs stacked on top of each other
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=None, mask_zero=True)
        # replace the initial weights of the `embedding` layer with the pre-trained embeddings.
        if embeddings is not None:
            self.embedding.set_weights([embeddings])
        
        # these are irrelevant for determining sentiment:
        self.embedding.weights[0][pad_idx].assign(tf.zeros(embedding_dim))
        self.embedding.weights[0][unk_idx].assign(tf.zeros(embedding_dim))
        
        self.rnn = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_dim, return_sequences=True), name='bi_lstm')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.fc = tf.keras.layers.Dense(output_dim, activation='softmax')

    def call(self, inputs, training=None):
        # text : [batch size, sent len]
        
        embedded = self.dropout(self.embedding(inputs), training=training)
        # embedded : [batch size, sent len, emb dim]

        # pack sequence
        lengths = tf.reduce_sum(tf.cast(inputs != 0, tf.int32), axis=1)
        packed_embedded = tf.keras.layers.Masking()(embedded)
        packed_output = self.rnn(packed_embedded)
        # output (packed_output) is the concatenation of the hidden state from every time step
        # hidden is simply the final hidden state.
        # packed_output : [batch size, sent len, hid dim * num directions]
        
        # concat the final forward (hidden[:, -2, :]) and backward (hidden[:, -1, :]) hidden layers
        hidden = self.dropout(tf.concat([packed_output[:, -1, :self.hidden_dim], packed_output[:, 0, self.hidden_dim:]], axis=1), training=training)
        # hidden : [batch size, hid dim * num directions]

        return self.fc(hidden)
