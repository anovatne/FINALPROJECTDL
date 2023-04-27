"""
Data loader for all parts
@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/tayebiarasteh/
https://tayebiarasteh.com/
"""

import numpy as np
import random
import torch
from torchtext import data
from Train_Test_Valid import Mode
from configs.serde import read_config
import os
import pandas as pd
from configs.serde import *
import pdb

epsilon = 1e-15


class data_provider_V2():
    '''
    Data handler class for the Standard Twitter sentiment classifier
    Packed padded sequences
    Tokenizer: spacy
    '''
    def __init__(self, cfg_path, batch_size=1, split_ratio=0.8, max_vocab_size=25000, mode=Mode.TRAIN, model_mode='RNN', seed=1):
        '''
        Args:
            cfg_path (string):
                Config file path of the experiment
            max_vocab_size (int):
                The number of unique words in our training set is usually over 100,000,
                which means that our one-hot vectors will have over 100,000 dimensions! SLOW TRAINIG!
                We only take the top max_vocab_size most common words.
            split_ratio (float):
                train-valid splitting
            mode (enumeration Mode):
                Nature of operation to be done with the data.
                Possible inputs are Mode.PREDICTION, Mode.TRAIN, Mode.VALID, Mode.TEST
                Default value: Mode.TRAIN
        '''
        params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.mode = mode
        self.seed = seed
        self.split_ratio = split_ratio
        self.max_vocab_size = max_vocab_size
        self.dataset_path = params['input_data_path']
        self.train_file_name = params['train_file_name']
        self.test_file_name = params['test_file_name']
        self.data_format = params['data_format']
        self.pretrained_embedding = params['pretrained_embedding']
        self.tokenizer = params['tokenizer']
        self.batch_size = batch_size
        self.model_mode = model_mode


    def data_loader(self):
        '''
        :include_lengths: Packed padded sequences: will make our RNN only process the non-padded elements of our sequence,
            and for any padded element the `output` will be a zero tensor.
            Note: padding is done by adding <pad> (not zero!)
        :tokenize: the "tokenization" (the act of splitting the string into discrete "tokens") should be done using the spaCy tokenizer.
        '''
        # if self.model_mode == 'RNN':
        #     #Packed padded sequences
        #     TEXT = data.Field(tokenize=self.tokenizer, include_lengths=True)  # For saving the length of sentences
        # if self.model_mode == 'CNN':
        #     TEXT = data.Field(tokenize=self.tokenizer, batch_first=True)  # batch dimension is the firs dimension here.
        # LABEL = data.LabelField()
        
        if self.model_mode == 'RNN':
        # Packed padded sequences
            TEXT = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='text')  # For variable-length sequences
            # LENGTH = tf.keras.layers.Input(shape=(), dtype=tf.int32, name='length')  # For saving the length of sentences
            LABEL = tf.keras.layers.Input(shape=(), dtype=tf.int32, name='label')
    
        # # Tokenize the input text
        #     tokenizer = Tokenizer()
        #     text_seq = tokenizer(TEXT)
    
        # # Pad the sequences to have the same length
        #     text_seq = pad_sequences(text_seq)
    
        # # Convert the length to a mask for the padded sequences
        #     mask = tf.sequence_mask(LENGTH)
    
        elif self.model_mode == 'CNN':
        # Batch dimension is the first dimension here
            TEXT = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='text')
            LABEL = tf.keras.layers.Input(shape=(), dtype=tf.int32, name='label')
    
        # # Tokenize the input text
        #     tokenizer = Tokenizer()
        #     text_seq = tokenizer(TEXT)
    
        # # Pad the sequences to have the same length
        #     text_seq = pad_sequences(text_seq, padding='post', truncating='post')

        fields = [('id', None), ('user_id', None),  ('label', LABEL), ('text', TEXT)]
        train_data, test_data = data.TabularDataset.splits(
            path=self.dataset_path,
            train=self.train_file_name,
            test=self.test_file_name,
            format=self.data_format,
            fields=fields,
            skip_header=False)

        # validation data
        if self.split_ratio == 1:
            valid_data = None
        else:
            train_data, valid_data = train_data.split(random_state=random.seed(self.seed), split_ratio=self.split_ratio)

        # create the vocabulary only on the training set!!!
        # vectors: instead of having our word embeddings initialized randomly, they are initialized with these pre-trained vectors.
        # initialize words in your vocabulary but not in your pre-trained embeddings to Gaussian
        TEXT.build_vocab(train_data, max_size=self.max_vocab_size,
                         vectors=self.pretrained_embedding, unk_init=torch.Tensor.normal_)
        LABEL.build_vocab(train_data)

        labels = LABEL.vocab.itos
        vocab_idx = TEXT.vocab.stoi

        vocab_size = len(TEXT.vocab)
        pretrained_embeddings = TEXT.vocab.vectors

        # the indices of the padding token <pad> and <unk> in the vocabulary
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
        # What do we do with words that appear in examples but we have cut from the vocabulary?
        # We replace them with a special unknown or <unk> token.

        # for packed padded sequences all of the tensors within a batch need to be sorted by their lengths
        # if self.split_ratio == 1:
        #     valid_iterator = None
        #     train_iterator, test_iterator = data.BucketIterator.splits((
        #         train_data, test_data), batch_size=self.batch_size,
        #         sort_within_batch=True, sort_key=lambda x: len(x.text))
        # else:
        #     train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((
        #         train_data, valid_data, test_data), batch_size=self.batch_size,
        #         sort_within_batch=True, sort_key=lambda x: len(x.text))
        
        if self.split_ratio == 1:
            valid_iterator = None
            train_data = train_data.shuffle(buffer_size=len(train_data))
            train_iterator = tf.data.Dataset.from_tensor_slices((train_data.text, train_data.label))
            train_iterator = train_iterator.batch(self.batch_size)
            train_iterator = train_iterator.prefetch(buffer_size=tf.data.AUTOTUNE)

            test_data = test_data.shuffle(buffer_size=len(test_data))
            test_iterator = tf.data.Dataset.from_tensor_slices((test_data.text, test_data.label))
            test_iterator = test_iterator.batch(self.batch_size)
            test_iterator = test_iterator.prefetch(buffer_size=tf.data.AUTOTUNE)
        else:
            train_data = train_data.shuffle(buffer_size=len(train_data))
            train_iterator = tf.data.Dataset.from_tensor_slices((train_data.text, train_data.label))
            train_iterator = train_iterator.batch(self.batch_size)
            train_iterator = train_iterator.prefetch(buffer_size=tf.data.AUTOTUNE)

            valid_data = valid_data.shuffle(buffer_size=len(valid_data))
            valid_iterator = tf.data.Dataset.from_tensor_slices((valid_data.text, valid_data.label))
    v       alid_iterator = valid_iterator.batch(self.batch_size)
            valid_iterator = valid_iterator.prefetch(buffer_size=tf.data.AUTOTUNE)

            test_data = test_data.shuffle(buffer_size=len(test_data))
            test_iterator = tf.data.Dataset.from_tensor_slices((test_data.text, test_data.label))
            test_iterator = test_iterator.batch(self.batch_size)
            test_iterator = test_iterator.prefetch(buffer_size=tf.data.AUTOTUNE)

        # finding the weights of each label
        data_for_weight = pd.read_csv(os.path.join(self.dataset_path, self.train_file_name), sep='\t',
                                      names=['id1', 'id2', 'label', 'tweet'])
        pos_counter = 0
        neg_counter = 0
        neut_counter = 0
        for i in range(len(data_for_weight['label'])):
            if (data_for_weight['label'][i] == 'positive'):
                pos_counter += 1
            if (data_for_weight['label'][i] == 'negative'):
                neg_counter += 1
            if (data_for_weight['label'][i] == 'neutral'):
                neut_counter += 1
        overall = neut_counter + pos_counter + neg_counter
        neut_weight = overall/neut_counter
        neg_weight = overall/neg_counter
        pos_weight = overall/pos_counter
        if labels == ['neutral', 'negative', 'positive']:
            weights = tf.constant([neut_weight, neg_weight, pos_weight])
        elif labels == ['neutral', 'positive', 'negative']:
            weights = tf.constant([neut_weight, pos_weight, neg_weight])
        elif labels == ['negative', 'neutral', 'positive']:
            weights = tf.constant([neg_weight, neut_weight, pos_weight])
        elif labels == ['negative', 'positive', 'neutral']:
            weights = tf.constant([neg_weight, pos_weight, neut_weight])
        elif labels == ['positive', 'negative', 'neutral']:
            weights = tf.constant([pos_weight, neg_weight, neut_weight])
        elif labels == ['positive', 'neutral', 'negative']:
            weights = tf.constant([pos_weight, neut_weight, neg_weight])

        if self.mode == Mode.TEST:
            return test_iterator, vocab_size, PAD_IDX, UNK_IDX, pretrained_embeddings, labels
        elif self.mode == Mode.PREDICTION:
            return labels, vocab_idx, vocab_size, PAD_IDX, UNK_IDX, pretrained_embeddings, labels
        else:
            return train_iterator, valid_iterator, vocab_size, PAD_IDX, UNK_IDX, pretrained_embeddings, weights, labels



class data_provider_PostReply():
    '''
    Packed padded sequences
    Tokenizer: spacy
    '''
    def __init__(self, cfg_path, batch_size=1, split_ratio=0.8, max_vocab_size=25000, mode=Mode.TRAIN, model_mode='RNN', seed=1):
        '''
        Args:
            cfg_path (string):
                Config file path of the experiment
            max_vocab_size (int):
                The number of unique words in our training set is usually over 100,000,
                which means that our one-hot vectors will have over 100,000 dimensions! SLOW TRAINIG!
                We only take the top max_vocab_size most common words.
            split_ratio (float):
                train-valid splitting
            mode (enumeration Mode):
                Nature of operation to be done with the data.
                Possible inputs are Mode.PREDICTION, Mode.TRAIN, Mode.VALID, Mode.TEST
                Default value: Mode.TRAIN
        '''
        params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.mode = mode
        self.seed = seed
        self.split_ratio = split_ratio
        self.max_vocab_size = max_vocab_size
        self.dataset_path = params['postreply_data_path']
        self.train_file_name = params['training_post_reply_file_name']
        self.test_file_name = params['final_test_post_reply_file_name']
        self.data_format = params['reply_data_format']
        self.pretrained_embedding = params['pretrained_embedding']
        self.tokenizer = params['tokenizer']
        self.batch_size = batch_size
        self.model_mode = model_mode


    def data_loader(self):
        '''
        :include_lengths: Packed padded sequences: will make our RNN only process the non-padded elements of our sequence,
            and for any padded element the `output` will be a zero tensor.
            Note: padding is done by adding <pad> (not zero!)
        :tokenize: the "tokenization" (the act of splitting the string into discrete "tokens") should be done using the spaCy tokenizer.
        '''
        # if self.model_mode == 'RNN':
        #     #Packed padded sequences
        #     TEXT = data.Field(tokenize=self.tokenizer, include_lengths=True)  # For saving the length of sentences
        # if self.model_mode == 'CNN':
        #     TEXT = data.Field(tokenize=self.tokenizer, batch_first=True)  # batch dimension is the firs dimension here.
        # LABEL = data.LabelField()
        
        if self.model_mode == 'RNN':
        # Packed padded sequences
            TEXT = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='text')  # For variable-length sequences
            # LENGTH = tf.keras.layers.Input(shape=(), dtype=tf.int32, name='length')  # For saving the length of sentences
            LABEL = tf.keras.layers.Input(shape=(), dtype=tf.int32, name='label')
    
        # # Tokenize the input text
        #     tokenizer = Tokenizer()
        #     text_seq = tokenizer(TEXT)
    
        # # Pad the sequences to have the same length
        #     text_seq = pad_sequences(text_seq)
    
        # # Convert the length to a mask for the padded sequences
        #     mask = tf.sequence_mask(LENGTH)
    
        elif self.model_mode == 'CNN':
        # Batch dimension is the first dimension here
            TEXT = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='text')
            LABEL = tf.keras.layers.Input(shape=(), dtype=tf.int32, name='label')
    
        # # Tokenize the input text
        #     tokenizer = Tokenizer()
        #     text_seq = tokenizer(TEXT)
    
        # # Pad the sequences to have the same length
        #     text_seq = pad_sequences(text_seq, padding='post', truncating='post')

        fields = [('label', LABEL), ('id', None), ('text', TEXT)]

        train_data, test_data = data.TabularDataset.splits(
            path=self.dataset_path,
            train=self.train_file_name,
            test=self.test_file_name,
            format=self.data_format,
            fields=fields,
            skip_header=True)

        # validation data
        if self.split_ratio == 1:
            valid_data = None
        else:
            train_data, valid_data = train_data.split(random_state=random.seed(self.seed), split_ratio=self.split_ratio)

        # create the vocabulary only on the training set!!!
        # vectors: instead of having our word embeddings initialized randomly, they are initialized with these pre-trained vectors.
        # initialize words in your vocabulary but not in your pre-trained embeddings to Gaussian
        TEXT.build_vocab(train_data, max_size=self.max_vocab_size,
                         vectors=self.pretrained_embedding, unk_init=torch.Tensor.normal_)
        # TEXT.build_vocab(train_data, max_size=self.max_vocab_size)

        LABEL.build_vocab(train_data)

        labels = LABEL.vocab.itos
        vocab_idx = TEXT.vocab.stoi

        vocab_size = len(TEXT.vocab)
        pretrained_embeddings = TEXT.vocab.vectors

        # the indices of the padding token <pad> and <unk> in the vocabulary
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
        # What do we do with words that appear in examples but we have cut from the vocabulary?
        # We replace them with a special unknown or <unk> token.


        # for packed padded sequences all of the tensors within a batch need to be sorted by their lengths
        # if self.split_ratio == 1:
        #     valid_iterator = None
        #     train_iterator, test_iterator = data.BucketIterator.splits((
        #         train_data, test_data), batch_size=self.batch_size,
        #         sort_within_batch=True, sort_key=lambda x: len(x.text))
        # else:
        #     train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((
        #         train_data, valid_data, test_data), batch_size=self.batch_size,
        #         sort_within_batch=True, sort_key=lambda x: len(x.text))
            
        if self.split_ratio == 1:
            valid_iterator = None
            train_data = train_data.shuffle(buffer_size=len(train_data))
            train_iterator = tf.data.Dataset.from_tensor_slices((train_data.text, train_data.label))
            train_iterator = train_iterator.shuffle(buffer_size=len(train_data))
            train_iterator = train_iterator.padded_batch(self.batch_size, padded_shapes=([None], []))
            train_iterator = train_iterator.prefetch(buffer_size=tf.data.AUTOTUNE)

            test_data = test_data.shuffle(buffer_size=len(test_data))
            test_iterator = tf.data.Dataset.from_tensor_slices((test_data.text, test_data.label))
            test_iterator = test_iterator.shuffle(buffer_size=len(test_data))
            test_iterator = test_iterator.padded_batch(self.batch_size, padded_shapes=([None], []))
            test_iterator = test_iterator.prefetch(buffer_size=tf.data.AUTOTUNE)
        else:
            train_data = train_data.shuffle(buffer_size=len(train_data))
            train_iterator = tf.data.Dataset.from_tensor_slices((train_data.text, train_data.label))
            train_iterator = train_iterator.shuffle(buffer_size=len(train_data))
            train_iterator = train_iterator.padded_batch(self.batch_size, padded_shapes=([None], []))
            train_iterator = train_iterator.prefetch(buffer_size=tf.data.AUTOTUNE)

            valid_data = valid_data.shuffle(buffer_size=len(valid_data))
            valid_iterator = tf.data.Dataset.from_tensor_slices((valid_data.text, valid_data.label))
            valid_iterator = valid_iterator.shuffle(buffer_size=len(valid_data))
            valid_iterator = valid_iterator.padded_batch(self.batch_size, padded_shapes=([None], []))
            valid_iterator = valid_iterator.prefetch(buffer_size=tf.data.AUTOTUNE)

            test_data = test_data.shuffle(buffer_size=len(test_data))
            test_iterator = tf.data.Dataset.from_tensor_slices((test_data.text, test_data.label))
            test_iterator = test_iterator.shuffle(buffer_size=len(test_data))
            test_iterator = test_iterator.padded_batch(self.batch_size, padded_shapes=([None], []))
            test_iterator = test_iterator.prefetch(buffer_size=tf.data.AUTOTUNE)

        # finding the weights of each label
        data_for_weight = pd.read_csv(os.path.join(self.dataset_path, self.train_file_name))
        pos_counter = 0
        neg_counter = 0
        neut_counter = 0
        for i in range(len(data_for_weight['label'])):
            if (data_for_weight['label'][i] == 'positive'):
                pos_counter += 1
            if (data_for_weight['label'][i] == 'negative'):
                neg_counter += 1
            if (data_for_weight['label'][i] == 'neutral'):
                neut_counter += 1
        overall = neut_counter + pos_counter + neg_counter
        neut_weight = overall/neut_counter
        neg_weight = overall/neg_counter
        pos_weight = overall/pos_counter
        if labels == ['neutral', 'negative', 'positive']:
            weights = tf.constant([neut_weight, neg_weight, pos_weight], dtype=tf.float32)
        elif labels == ['neutral', 'positive', 'negative']:
            weights = tf.constant([neut_weight, pos_weight, neg_weight], dtype=tf.float32)
        elif labels == ['negative', 'neutral', 'positive']:
            weights = tf.constant([neg_weight, neut_weight, pos_weight], dtype=tf.float32)
        elif labels == ['negative', 'positive', 'neutral']:
            weights = tf.constant([neg_weight, pos_weight, neut_weight], dtype=tf.float32)
        elif labels == ['positive', 'negative', 'neutral']:
            weights = tf.constant([pos_weight, neg_weight, neut_weight], dtype=tf.float32)
        elif labels == ['positive', 'neutral', 'negative']:
            weights = tf.constant([pos_weight, neut_weight, neg_weight], dtype=tf.float32)

        if self.mode == Mode.TEST:
            return test_iterator
        elif self.mode == Mode.PREDICTION:
            return labels, vocab_idx, vocab_size, PAD_IDX, UNK_IDX, pretrained_embeddings, labels
        else:
            return train_iterator, valid_iterator, vocab_size, PAD_IDX, UNK_IDX, pretrained_embeddings, weights, labels




if __name__=='__main__':
    CONFIG_PATH = '../configs/config.json'
    data_handler = data_provider_PostReply(cfg_path=CONFIG_PATH, batch_size=1, split_ratio=0.8, max_vocab_size=25000)
    train_iterator, valid_iterator, vocab_size, PAD_IDX, UNK_IDX, pretrained_embeddings, classes = data_handler.data_loader()
    # pdb.set_trace()
