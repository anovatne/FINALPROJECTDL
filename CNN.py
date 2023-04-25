"""
CNN model for the text data.

"""

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import tensorflow as tf # added
import pdb


class CNN1d(tf.keras.layers.Layer): # nn.Module --> tf.keras.layers.Layer
    def __init__(self, vocab_size, embeddings, embedding_dim=200,
                 conv_out_ch=200, filter_sizes=[3,4,5], output_dim=3, pad_idx=1, unk_idx=0):
        '''
        :pad_idx: the index of the padding token <pad> in the vocabulary
        :conv_out_ch: number of the different kernels.
        :filter_sizes: a list of different kernel sizes we use here.
        '''
        super(CNN1d, self).__init__() # minor line change here
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        
        # replace the initial weights of the `embedding` layer with the pre-trained embeddings.
        # self.embedding.weight.data.copy_(embeddings)
        self.embedding.set_weights([tf.Variable(embeddings)]) # not sure
        
        # these are irrelevant for determining sentiment:
        # self.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)
        # self.embedding.weight.data[unk_idx] = torch.zeros(embedding_dim)
        self.embedding.weights[pad_idx].assign(tf.zeros(embedding_dim)) # not sure about this
        self.embedding.weights[unk_idx].assign(tf.zeros(embedding_dim))


        self.convc = tf.keras.Sequential(
            layers=[
            tf.keras.layers.Conv1D(filters=conv_out_ch, kernel_size=fs, activation='relu') for fs in filter_sizes
            ]
        )
        # self.convs = tf.keras.Sequential([
        #     nn.Conv1d(in_channels=embedding_dim, out_channels=conv_out_ch,
        #               kernel_size=fs) for fs in filter_sizes])

        self.fc = tf.keras.layers.Dense(output_dim)
        # self.fc = nn.Linear(len(filter_sizes) * conv_out_ch, output_dim)
        
        self.dropout = tf.keras.layers.Dropout(0.5)
        # self.dropout = nn.Dropout(0.5)


    def call(self, text):
        # text = [batch size, sent len]

        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]

        embedded = tf.transpose(embedded, perm=[0, 2, 1])
        # embedded = embedded.permute(0, 2, 1)
        # embedded = [batch size, emb dim, sent len]

        # pad if the length of the sentence is less than the kernel size
        if embedded.shape[2] < 5:
            dif = 5 - embedded.shape[2]
            embedded = tf.pad(embedded, [[0, 0], [0, 0], [0, dif]], constant_values=0.0)
            # embedded = F.pad(embedded, (0, dif), "constant", 0)

        conved = [tf.nn.relu(conv(embedded)) for conv in self.convs]
        # conved = [F.relu(conv(embedded)) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [tf.reduce_max(conv, axis=2) for conv in conved]
        # pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]

        cat = self.dropout(tf.concat(pooled, axis=1))
        # cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)