import tensorflow as tf

class biLSTM(tf.keras.Model):
    def __init__(self, vocab_size, embeddings=None, embedding_dim=100, hidden_dim=256, output_dim=3, pad_idx=1, unk_idx=0):
        super().__init__()

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=None, mask_zero=True)
        self.embedding.build((None,))
        # replace the initial weights of the `embedding` layer with the pre-trained embeddings.
        self.embedding.set_weights([embeddings])
        # these are irrelevant for determining sentiment:
        self.embedding.weights[0][pad_idx].assign(tf.zeros(embedding_dim))
        self.embedding.weights[0][unk_idx].assign(tf.zeros(embedding_dim))

        self.rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_dim, return_sequences=False, dropout=0.5), name="biLSTM")
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.fc = tf.keras.layers.Dense(output_dim, activation='softmax')

    def call(self, text, text_lengths):
        # embedded : [batch size, sent len, emb dim]
        embedded = self.dropout(self.embedding(text))

        # outputs : [batch size, sent len, hid dim * num directions]
        # state_h, state_c : [num layers * num directions, batch size, hid dim]
        outputs, state_h, state_c = self.rnn(embedded)

        # concatenate the final forward and backward hidden states
        # hidden : [batch size, hid dim * num directions]
        hidden = self.dropout(tf.concat([state_h[-2,:,:], state_h[-1,:,:]], axis=1))

        # output : [batch size, output dim]
        output = self.fc(hidden)
        return output
