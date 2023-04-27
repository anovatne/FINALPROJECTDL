import os
from enum import Enum
import datetime
import time
import spacy
import matplotlib.pyplot as plt
import itertools

import numpy as np
import tensorflow as tf
from sklearn import metrics
from enum import Enum

# User Defined Modules
from configs.serde import *
from models.biLSTM import *
from models.CNN import *

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Training:
    '''
    This class represents training process.
    '''

    def __init__(self, cfg_path, num_epochs=10, RESUME=False, model_mode='RNN', tf_seed=None):
        '''
        :cfg_path (string): path of the experiment config file
        :tf_seed (int): Seed used for random generators in TensorFlow functions
        '''
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.RESUME = RESUME
        self.model_mode = model_mode
        self.num_epochs = num_epochs

        if RESUME == False:
            self.model_info = self.params['Network']
            self.model_info['seed'] = tf_seed or self.model_info['seed']
            self.epoch = 0
            self.num_epochs = num_epochs
            self.best_loss = float('inf')
            if 'trained_time' in self.model_info:
                self.raise_training_complete_exception()
            self.setup_seed()

    def setup_seed(self):
        np.random.seed(self.model_info['seed'])
        tf.random.set_seed(self.model_info['seed'])

    def setup_model(self, model, optimiser, loss_function, weight):

        total_param_num = np.sum([tf.size(w).numpy() for w in model.trainable_variables])
        print(f"Total # of model's trainable parameters: {total_param_num:,}")
        print('----------------------------------------------------\n')

        self.model = model
        self.optimiser = optimiser
        self.loss_function = loss_function

        if 'retrain' in self.model_info and self.model_info['retrain'] == True:
            self.load_pretrained_model()

        # Saves the model, optimiser, loss function name for writing to config file
        self.model_info['total_param_num'] = total_param_num
        self.model_info['optimiser'] = optimiser._name
        self.model_info['loss_function'] = loss_function.__name__
        self.params['Network'] = self.model_info
        write_config(self.params, self.cfg_path, sort_keys=True)

    def load_checkpoint(self, model, optimiser, loss_function, weight):

        checkpoint = tf.train.latest_checkpoint(self.params['network_output_path'])
        ckpt = tf.train.Checkpoint(model=model, optimiser=optimiser)
        ckpt.restore(checkpoint).expect_partial()

        self.model = model
        self.optimiser = optimiser
        self.loss_function = loss_function

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    
    
    def execute_training(self, train_dataset, valid_dataset=None, batch_size=1):
        '''
        Executes training by running training and validation at each epoch
        '''
        self.params = read_config(self.cfg_path)

        total_start_time = time.time()

        if self.RESUME == False:
            # Checks if already trained
            if 'trained_time' in self.model_info:
                self.raise_training_complete_exception

            self.model_info = self.params['Network']
            self.model_info['num_epoch'] = self.num_epochs or self.model_info['num_epoch']

        print('Starting time:' + str(datetime.datetime.now()) +'\n')

        for epoch in range(self.num_epochs - self.epoch):
            self.epoch += 1
            start_time = time.time()

            print('Training (intermediate metrics):')
            train_loss, train_acc, train_F1, train_recall, train_precision = self.train_epoch(train_dataset, batch_size)

            if valid_dataset:
                print('\nValidation (intermediate metrics):')
                valid_loss, valid_acc, valid_F1, valid_recall, valid_precision = self.valid_epoch(valid_dataset, batch_size)

            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
            total_mins, total_secs = self.epoch_time(total_start_time, end_time)

            # Saving the model
            if valid_dataset:
                if valid_loss < self.best_loss:
                    self.best_loss = valid_loss
                    self.model.save(self.params['network_output_path'] + '/' + self.params['trained_model_name'])
            else:
                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self.model.save(self.params['network_output_path'] + '/' + self.params['trained_model_name'])

            # Saving the model based on epoch, checkpoint
            self.savings()

            # Print accuracy, F1, and loss after each epoch
            print('\n---------------------------------------------------------------')
            print(f'Epoch: {self.epoch:02} | Epoch Time: {epoch_mins}m {epoch_secs}s | '
                f'Total Time so far: {total_mins}m {total_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Train F1: {train_F1:.3f}')
            if valid_dataset:
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}% |  Val. F1: {valid_F1:.3f}')
            print('---------------------------------------------------------------\n')
            


    def train_epoch(self, train_dataset, batch_size):
        '''
        Train using one single iteration of all messages (epoch) in dataset
        '''
        print("Epoch [{}/{}]".format(self.epoch, self.model_info['num_epoch']))
        self.model.train()

        # initializing the loss list
        batch_loss = 0
        batch_count = 0

        # initializing the caches
        logits_cache = np.zeros((len(train_dataset) * batch_size, 3))
        max_preds_cache = np.zeros((len(train_dataset) * batch_size, 1))
        labels_cache = np.zeros(len(train_dataset) * batch_size)

        for idx, (message, label) in enumerate(train_dataset.batch(batch_size)):
            with tf.GradientTape() as tape:
                if self.model_mode == "RNN":
                    output = self.model([message, tf.math.reduce_sum(tf.cast(tf.math.not_equal(message, 0), tf.int32), axis=1)])
                if self.model_mode == "CNN":
                    output = self.model(message)

                # Loss
                loss = self.loss_function(label, output)
                batch_loss += loss.numpy()
                batch_count += 1
                max_preds = tf.argmax(output, axis=1, output_type=tf.int32, keepdims=True)  # get the index of the max probability

                # saving the logits and labels of this batch
                logits_cache[idx * batch_size:(idx + 1) * batch_size] = output.numpy()
                max_preds_cache[idx * batch_size:(idx + 1) * batch_size] = max_preds.numpy()
                labels_cache[idx * batch_size:(idx + 1) * batch_size] = label.numpy()

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimiser.apply_gradients(zip(gradients, self.model.trainable_variables))

            # Prints loss statistics after number of steps specified.
            if (idx + 1) % self.params['display_stats_freq'] == 0:
                print('Epoch {:02} | Batch {:03}-{:03} | Train loss: {:.3f}'.
                    format(self.epoch, idx - self.params['display_stats_freq'] + 1, idx, batch_loss / batch_count))
                batch_loss = 0
                batch_count = 0

        '''Metrics calculation over the whole set'''
        # average=None gives individual scores for each class
        # here we only care about the average of positive class and negative class
        epoch_accuracy = metrics.accuracy_score(labels_cache, max_preds_cache)
        epoch_f1_score = metrics.f1_score(labels_cache, max_preds_cache, average=None)
        epoch_precision = metrics.precision_score(labels_cache, max_preds_cache, average=None)
        epoch_recall = metrics.recall_score(labels_cache, max_preds_cache, average=None)
        epoch_f1_score = (epoch_f1_score[1] + epoch_f1_score[2]) / 2
        epoch_precision = (epoch_precision[1] + epoch_precision[2]) / 2
        epoch_recall = (epoch_recall[1] + epoch_recall[2]) / 2

        # Loss
        loss = self.loss_function(labels_cache, logits_cache)
        epoch_loss = loss.numpy()

        return epoch_loss, epoch_accuracy, epoch_f1_score, epoch_precision, epoch_recall

    def valid_epoch(self, valid_dataset, batch_size):
        '''Test (validation) model after an epoch and calculate loss on valid dataset'''
        print("Epoch [{}/{}]".format(self.epoch, self.model_info['num_epoch']))
        self.model.evaluate()

        # initializing the loss list
        batch_loss = 0
        batch_count = 0

        # initializing the caches
        logits_cache = np.zeros((len(valid_dataset) * batch_size, 3))
        max_preds_cache = np.zeros((len(valid_dataset) * batch_size, 1))
        labels_cache = np.zeros(len(valid_dataset) * batch_size)

        for idx, (message, label) in enumerate(valid_dataset.batch(batch_size)):
            if self.model_mode == "RNN":
                output = self.model([message, tf.math.reduce_sum(tf.cast(tf.math.not_equal(message, 0), tf.int32), axis=1)])
            if self.model_mode == "CNN":
                output = self.model(message)

            # Loss
            loss = self.loss_function(label, output)
            batch_loss += loss.numpy()
            batch_count += 1
            max_preds = tf.argmax(output, axis=1, output_type=tf.int32, keepdims=True)  # get the index of the max probability

            # saving the logits and labels of this batch
            logits_cache[idx * batch_size:(idx + 1) * batch_size] = output.numpy()
            max_preds_cache[idx * batch_size:(idx + 1) * batch_size] = max_preds.numpy()
            labels_cache[idx * batch_size:(idx + 1) * batch_size] = label.numpy()

            # Prints loss statistics after number of steps specified.
            if (idx + 1) % self.params['display_stats_freq'] == 0:
                print('Epoch {:02} | Batch {:03}-{:03} | Val. loss: {:.3f}'.
                    format(self.epoch, idx - self.params['display_stats_freq'] + 1, idx, batch_loss / batch_count))
                batch_loss = 0
                batch_count = 0

        '''Metrics calculation over the whole set'''
        epoch_accuracy = metrics.accuracy_score(labels_cache, max_preds_cache)
        epoch_f1_score = metrics.f1_score(labels_cache, max_preds_cache, average=None)
        epoch_precision = metrics.precision_score(labels_cache, max_preds_cache, average=None)
        epoch_recall = metrics.recall_score(labels_cache, max_preds_cache, average=None)
        epoch_f1_score = (epoch_f1_score[1] + epoch_f1_score[2]) / 2
        epoch_precision = (epoch_precision[1] + epoch_precision[2]) / 2
        epoch_recall = (epoch_recall[1] + epoch_recall[2]) / 2

        # Loss
        loss = self.loss_function(labels_cache, logits_cache)
        epoch_loss = loss.numpy()

        self.model.train()
        return epoch_loss, epoch_accuracy, epoch_f1_score, epoch_precision, epoch_recall

    def savings(self):
        # Saves information about training to config file
        self.model_info['num_steps'] = self.epoch
        self.model_info['trained_time'] = "{:%B %d, %Y, %H:%M:%S}".format(datetime.datetime.now())
        self.params['Network'] = self.model_info
        write_config(self.params, self.cfg_path, sort_keys=True)

        # Saving every 5 epochs
        if (self.epoch) % self.params['network_save_freq'] == 0:
            self.model.save_weights(self.params['network_output_path'] + '/' +
                                    'epoch{}_'.format(self.epoch) + self.params['trained_model_name'])

        # Save a checkpoint every epoch
        checkpoint = {
            'epoch': self.epoch,
            'model_weights': self.model.get_weights(),
            'optimizer_weights': self.optimiser.get_weights(),
            'loss': self.loss_function, 'num_epoch': self.num_epochs,
            'model_info': self.model_info, 'best_loss': self.best_loss
        }
        with open(self.params['network_output_path'] + '/' + self.params['checkpoint_name'], 'wb') as f:
            pickle.dump(checkpoint, f)

    def calculate_tb_stats(self, train_loss, train_F1, train_recall, train_precision, train_accuracy,
                        valid_loss=None, valid_F1=None, valid_recall=None, valid_precision=None, valid_accuracy=None):

        # Adds the metrics to TensorBoard
        self.writer.add_scalar('Training' + '_Loss', train_loss, self.epoch)
        self.writer.add_scalar('Training' + '_F1', train_F1, self.epoch)
        self.writer.add_scalar('Training' + '_Recall', train_recall, self.epoch)
        self.writer.add_scalar('Training' + '_Precision', train_precision, self.epoch)
        self.writer.add_scalar('Training' + '_Accuracy', train_accuracy, self.epoch)
        if valid_loss:
            self.writer.add_scalar('Validation' + '_Loss', valid_loss, self.epoch)
            self.writer.add_scalar('Validation' + '_F1', valid_F1, self.epoch)
            self.writer.add_scalar('Validation' + '_Recall', valid_recall, self.epoch)
            self.writer.add_scalar('Validation' + '_Precision', valid_precision, self.epoch)
            self.writer.add_scalar('Validation' + '_Accuracy', valid_accuracy, self.epoch)

    def load_pretrained_model(self):
        '''Load pre trained model to the using pre-trained_model_path parameter from config file'''
        self.model.load_weights(self.model_info['pretrain_model_path'])

    def raise_training_complete_exception(self):
        raise Exception("Model has already been trained on {}. \n"
                        "1.To use this model as pre trained model and train again\n "
                        "create new experiment using create_retrain_experiment function.\n\n"
                        "2.To start fresh with same experiment name, delete the experiment  \n"
                        "using delete_experiment function and create experiment "
                        "               again.".format(self.model_info['trained_time']))

class Prediction:
    '''
    This class represents prediction (testing) process similar to the Training class.
    '''
    def __init__(self, cfg_path, classes, model_mode='RNN', cfg_path_RNN=None, cfg_path_CNN=None):
        self.params = read_config(cfg_path)
        if cfg_path_CNN:
            self.params_RNN = read_config(cfg_path_RNN)
            self.params_CNN = read_config(cfg_path_CNN)
        self.cfg_path = cfg_path
        self.setup_cuda()
        self.model_mode = model_mode
        self.classes = classes

    def setup_cuda(self, cuda_device_id=0):
        self.device = tf.device('GPU:{}'.format(cuda_device_id)) if tf.config.list_physical_devices('GPU') else tf.device('CPU:0')

    @staticmethod
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def setup_model(self, model, vocab_size, embeddings, embedding_dim,
                    hidden_dim, pad_idx, unk_idx, model_file_name=None, epoch=19,
                    conv_out_ch=200, filter_sizes=[3,4,5], model_c =CNN1d, model_r=biLSTM):
        if model_file_name == None:
            model_file_name = self.params['trained_model_name']
        if self.model_mode == "RNN":
            self.model_p = model(vocab_size=vocab_size, embeddings=embeddings, embedding_dim=embedding_dim,
                                 hidden_dim=hidden_dim, pad_idx=pad_idx, unk_idx=unk_idx)
        elif self.model_mode == "CNN":
            self.model_p = model(vocab_size=vocab_size, embeddings=embeddings, embedding_dim=embedding_dim,
                                 conv_out_ch=conv_out_ch, filter_sizes=filter_sizes, pad_idx=pad_idx, unk_idx=unk_idx)
        elif self.model_mode == "ensemble":
            model_file_name_c = self.params_CNN['trained_model_name']
            model_file_name_r = self.params_RNN['trained_model_name']
            self.model_cnn = model_c(vocab_size=vocab_size, embeddings=embeddings, embedding_dim=embedding_dim,
                                 conv_out_ch=conv_out_ch, filter_sizes=filter_sizes, pad_idx=pad_idx, unk_idx=unk_idx)
            self.model_rnn = model_r(vocab_size=vocab_size, embeddings=embeddings, embedding_dim=embedding_dim,
                                 hidden_dim=hidden_dim, pad_idx=pad_idx, unk_idx=unk_idx)

        # Loads model from model_file_name and default network_output_path
        if self.model_mode == "ensemble":
            self.model_cnn.load_weights(self.params_CNN['network_output_path'] + "/epoch" + str(19) + "_" + model_file_name_c)
            self.model_rnn.load_weights(self.params_RNN['network_output_path'] + "/epoch" + str(43) + "_" + model_file_name_r)
        else:
            self.model_p.load_weights(self.params['network_output_path'] + "/epoch" + str(epoch) + "_" + model_file_name)

    def predict(self, test_loader, batch_size):
        # Reads params to check if any params have been changed by user
        self.params = read_config(self.cfg_path)

    def predict(self, test_loader, batch_size):
        # Reads params to check if any params have been changed by user
        self.params = read_config(self.cfg_path)
        
        start_time = time.time()
        
        # initializing the caches
        logits_cache = np.zeros((len(test_loader) * batch_size, 3))
        max_preds_cache = np.zeros((len(test_loader) * batch_size, 1))
        labels_cache = np.zeros(len(test_loader) * batch_size)
        
        for idx, batch in enumerate(test_loader):
            if self.model_mode == "RNN":
                message, message_lengths = batch.text
            if self.model_mode == "CNN":
                message = batch.text
            label = batch.label
            message = message.astype(np.int64)
            label = label.astype(np.int64)
            
            if self.model_mode == "RNN":
                output = self.model_p(message, message_lengths).numpy().squeeze(1)
            if self.model_mode == "CNN":
                output = self.model_p(message).numpy().squeeze(1)
            
            max_preds = np.argmax(output, axis=1, keepdims=True)  # get the index of the max probability
            
            # saving the logits and labels of this batch
            max_preds_cache[idx * batch_size: idx * batch_size + len(max_preds)] = max_preds
            logits_cache[idx * batch_size: idx * batch_size + len(output)] = output
            labels_cache[idx * batch_size: idx * batch_size + len(label)] = label

        '''Metrics calculation over the whole set'''
        final_accuracy = metrics.accuracy_score(labels_cache, max_preds_cache)
        final_f1_score = metrics.f1_score(labels_cache, max_preds_cache, average=None)
        final_precision = metrics.precision_score(labels_cache, max_preds_cache, average=None)
        final_recall = metrics.recall_score(labels_cache, max_preds_cache, average=None)
        final_f1_score = (final_f1_score[1] + final_f1_score[2]) / 2
        final_precision = (final_precision[1] + final_precision[2]) / 2
        final_recall = (final_recall[1] + final_recall[2]) / 2
        confusion_matrix = metrics.confusion_matrix(labels_cache, max_preds_cache, labels=[0, 1, 2])
        
        end_time = time.time()
        test_mins, test_secs = self.epoch_time(start_time, end_time)
        
        # Print the final evaluation metrics
        print('\n----------------------------------------------------------------------')
        print(f'Testing | Testing Time: {test_mins}m {test_secs}s')
        print(f'\tAcc: {final_accuracy * 100:.2f}% | F1 score: {final_f1_score:.3f} | '
            f'Recall: {final_recall:.3f} | Precision: {final_precision:.3f}')
        print('----------------------------------------------------------------------\n')
        print(confusion_matrix)
        # self.plot_confusion_matrix(confusion_matrix, target_names=self.classes,
        #                       title='Confusion matrix, without normalization')
        return final_accuracy, final_f1_score

        
    def predict_ensemble(self, test_iterator_RNN, test_iterator_CNN, batch_size):
        "prediction with ensembling CNN and RNN outputs by normal averaging"

        # Reads params to check if any params have been changed by user
        self.params = read_config(self.cfg_path)
        self.model_cnn.eval()
        self.model_rnn.eval()

        start_time = time.time()

        # initializing the caches
        logits_cache = np.zeros((len(test_iterator_RNN) * batch_size, 3))
        max_preds_cache = np.zeros((len(test_iterator_RNN) * batch_size, 1))
        labels_cache = np.zeros(len(test_iterator_RNN) * batch_size)

        for idx, (batch_RNN, batch_CNN) in enumerate(zip(test_iterator_RNN, test_iterator_CNN)):

            # RNN part
            message, message_lengths = batch_RNN.text
            label = batch_RNN.label
            message = message.astype(np.int32)
            label = label.astype(np.int32)
            message = tf.convert_to_tensor(message)
            label = tf.convert_to_tensor(label)
            output_RNN = tf.squeeze(self.model_rnn(message, message_lengths), axis=1)

            # CNN part
            message = batch_CNN.text
            label = batch_CNN.label
            message = message.astype(np.int32)
            label = label.astype(np.int32)
            message = tf.convert_to_tensor(message)
            label = tf.convert_to_tensor(label)
            output_CNN = tf.squeeze(self.model_cnn(message), axis=1)

            output = (output_CNN + output_RNN) / 2
            max_preds = tf.argmax(output, axis=1, output_type=tf.int32, keepdims=True)  # get the index of the max probability
            # saving the logits and labels of this batch
            for i, batch_vector in enumerate(max_preds.numpy()):
                max_preds_cache[idx * batch_size + i] = batch_vector
            for i, batch_vector in enumerate(output.numpy()):
                logits_cache[idx * batch_size + i] = batch_vector
            for i, value in enumerate(label.numpy()):
                labels_cache[idx * batch_size + i] = value

        '''Metrics calculation over the whole set'''

        # average=None gives individual scores for each class
        # here we only care about the average of positive class and negative class
        final_accuracy = metrics.accuracy_score(labels_cache, max_preds_cache)
        final_f1_score = metrics.f1_score(labels_cache, max_preds_cache, average=None)
        final_precision = metrics.precision_score(labels_cache, max_preds_cache, average=None)
        final_recall = metrics.recall_score(labels_cache, max_preds_cache, average=None)
        final_f1_score = (final_f1_score[1] + final_f1_score[2]) / 2
        final_precision = (final_precision[1] + final_precision[2]) / 2
        final_recall = (final_recall[1] + final_recall[2]) / 2
        confusion_matrix = metrics.confusion_matrix(labels_cache, max_preds_cache, labels=[0,1,2])

        end_time = time.time()
        test_mins, test_secs = self.epoch_time(start_time, end_time)

        # Print the final evaluation metrics
        print('\n----------------------------------------------------------------------')
        print(f'Testing | Testing Time: {test_mins}m {test_secs}s')
        print(f'\tAcc: {final_accuracy * 100:.2f}% | F1 score: {final_f1_score:.3f} | '
                f'Recall: {final_recall:.3f} | Precision: {final_precision:.3f}')
        print('----------------------------------------------------------------------\n')
        print(confusion_matrix)
        # self.plot_confusion_matrix(confusion_matrix, target_names=self.classes,
        #                       title='Confusion matrix, without normalization')
        return final_accuracy, final_f1_score

    def plot_confusion_matrix(self, cm, target_names,
                                title='Confusion matrix', cmap=None, normalize=False):
            """
            given a sklearn confusion matrix (cm), make a nice plot
            ---------
            cm:           confusion matrix from sklearn.metrics.confusion_matrix
            target_names: given classification classes such as [0, 1, 2]
                        the class names, for example: ['high', 'medium', 'low']
            cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                        plt.get_cmap('jet') or plt.cm.Blues
            normalize:    If False, plot the raw numbers
                        If True, plot the proportions
            """
            accuracy = np.trace(cm) / np.sum(cm).astype('float')
            misclass = 1 - accuracy

            if cmap is None:
                cmap = plt.get_cmap('Blues')

            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()

            if target_names is not None:
                tick_marks = np.arange(len(target_names))
                plt.xticks(tick_marks, target_names, rotation=45)
                plt.yticks(tick_marks, target_names)

            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            thresh = cm.max() / 1.5 if normalize else cm.max() / 2
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                if normalize:
                    plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")
                else:
                    plt.text(j, i, "{:,}".format(cm[i, j]),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label\naccuracy={:0.2f}%; misclass={:0.2f}%'.format(accuracy*100, misclass*100))
            plt.show()
        

    def manual_predict(self, labels, vocab_idx, phrase, min_len=4,
                    tokenizer=spacy.load('en'), mode=None, prediction_mode='Manualpart1'):
        '''
        Manually predicts the polarity of the given sentence.
        Possible polarities: 1.neutral, 2.positive, 3.negative
        '''
        self.params = read_config(self.cfg_path)
        self.model_p.trainable = False

        tokenized = [tok.text for tok in tokenizer.tokenizer(phrase)]
        if len(tokenized) < min_len:
            tokenized += ['<pad>'] * (min_len - len(tokenized))
        indexed = [vocab_idx[t] for t in tokenized]
        tensor = tf.convert_to_tensor(indexed, dtype=tf.int32)
        tensor = tf.expand_dims(tensor, axis=1)
        preds = self.model_p(tensor, tf.constant([tensor.shape[0]], dtype=tf.int32))
        max_preds = tf.argmax(preds, axis=1)
        if mode == Mode.REPLYPREDICTION:
            return labels[max_preds.numpy().item()]

        print('\n\t', '"' + phrase + '"')
        print('-----------------------------------------')
        if prediction_mode == 'Manualpart1':
            print(f'\t This is a {labels[max_preds.numpy().item()]} phrase!')
        elif prediction_mode == 'Manualpart2':
            print(f'\t This phrase is likely to get {labels[max_preds.numpy().item()]} replies!')
        print('-----------------------------------------')
        
        
class Mode(Enum):
    '''
    Class Enumerating the 3 modes of operation of the network.
    This is used while loading datasets
    '''
    TRAIN = 0
    VALID = 1
    TEST = 2
    PREDICTION = 3
    REPLYPREDICTION = 4
