import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import time
import random as rn

import tensorflow as tf
from tensorflow import keras
from keras import models, layers, initializers, regularizers, optimizers
from keras import callbacks as kc
from keras import backend as K

from sklearn.model_selection import KFold
from sklearn.metrics import log_loss


class GAPModel(object):


    def __init__(self):
        self.init_seed(1122)


    def init_seed(self, seed):
        """ Set various seeds for the result reproducibility """

        self.seed = seed
        os.environ['PYTHONHASHSEED'] = '0'
        np.random.seed(seed)
        rn.seed(seed)
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        tf.set_random_seed(seed)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)


    def load_embeddings(self, filename, data, idx_map):
        """
        Read a .json file with the BERT embeddings.
        Returns:
            data['df_ids'] - list of a DataFrame ID
            data['emb'] - list of the targets embeddings. Each item contains a list:
                0 - Pronoun embeddings
                1 - A-term embeddings
                2 - B-term embeddings
            idx_map - map a DataFrame ID into a list index.
        """

        with open(filename) as f:
            for line in tqdm(f):
                sample = json.loads(line)
                # Read the Pronoun embeddings only
                if sample['segment'] > 0:
                    continue
                layers = []
                # Concatenate all the output layer embeddings.
                # We wave 3*1024 values for each target. 
                for layer in sample['embeddings']:
                    layers += layer['values']
                layers = np.array(layers)
                df_idx = sample['df_idx']
                idx_map[df_idx] = len(data['emb'])
                data['df_ids'].append(df_idx)
                data['emb'].append([layers, np.zeros(layers.shape), np.zeros(layers.shape)])

        with open(filename) as f:
            for line in tqdm(f):
                sample = json.loads(line)
                # Read the A-term and B-term embeddings.
                if sample['segment'] == 0:
                    continue
                layers = []
                for layer in sample['embeddings']:
                    layers += layer['values']
                df_idx = sample['df_idx']
                segment = sample['segment']
                data['emb'][idx_map[df_idx]][segment] = np.array(layers)

        return data, idx_map


    def load_feats(self, filename, ids_filename, data, idx_map):
        """
        Reads a features .tsv file. Returns a list of the features.
        """
        feature_idx_map = []
        with open(ids_filename) as f:
            f.readline()
            for line in f:
                feature_idx_map.append(line.strip().split('\t')[0])

        with open(filename) as f:
            f.readline()
            for i, line in enumerate(f):
                arr = line.strip().split('\t')
                feats = [float(x) for x in arr[1:]]
                idx = feature_idx_map[int(arr[0])]
                data[idx_map[idx]] = feats

        return data


    def load_labels(self, filename, data, idx_map):
        """
        Reads an input GAP file. Returns a list of the labels:
        0 - for the 'NEITHER' label
        1 - for the 'A' label
        2 - for the 'B' label
        """
        df = pd.read_csv(filename, sep='\t')
        df['label'] = 0
        if 'A-coref-fixed' in df.columns:
            df.loc[df['A-coref-fixed'] == True, 'label'] = 1
            df.loc[df['B-coref-fixed'] == True, 'label'] = 2
        elif 'A-coref' in df.columns:
            df.loc[df['A-coref'] == True, 'label'] = 1
            df.loc[df['B-coref'] == True, 'label'] = 2
        # OHE of the labels
        for i, row in df.iterrows():
            idx = row.ID
            data[idx_map[idx]][row['label']] = 1
        return data


    def get_model(self, input_shapes):
        """ Main model function. """
        feat_shape = input_shapes[0] # Features number
        emb_shapes = input_shapes[1:] # Embeddings shapes

        def build_mlp_model(input_shape, model_num=0, dense_layer_size=128, dropout_rate=0.9):
            """
            The base MLP module. Dense + BatchNorm + Dropout.
            Idea from Matei Ionita's kernel:
            https://www.kaggle.com/mateiionita/taming-the-bert-a-baseline
            """
            X_input = layers.Input([input_shape])
            X = layers.Dense(dense_layer_size, name = 'mlp_dense_{}'.format(model_num), kernel_initializer=initializers.glorot_uniform(seed=self.seed))(X_input)
            X = layers.BatchNormalization(name = 'mlp_bn_{}'.format(model_num))(X)
            X = layers.Activation('relu')(X)
            X = layers.Dropout(dropout_rate, seed = self.seed)(X)
            model = models.Model(inputs = X_input, outputs = X, name = 'mlp_model_{}'.format(model_num))
            return model

        # Two models with inputs of shape (, 3*3*1014)
        # from BERT Large cased and BERT Large uncased embeddings.
        # Output shape is (, 112).
        all_models = [build_mlp_model(emb_shape, model_num=i, dense_layer_size=112, dropout_rate=0.9) for i, emb_shape in enumerate(emb_shapes)]

        # Two Siamese models with distances between 
        # Pronoun and A-term embeddings, 
        # Pronoun and B-term embeddings as inputs and shared weights. 
        # Input shape is (, 3*1024). Output shape is (, 2*112).
        for i, emb_shape in enumerate(emb_shapes):
            split_input = layers.Input([emb_shape])
            split_model_shape = int(emb_shape / 3)
            split_model = build_mlp_model(split_model_shape, model_num=len(all_models), dense_layer_size=112)
            P = layers.Lambda(lambda x: x[:, :split_model_shape])(split_input)
            A = layers.Lambda(lambda x: x[:, split_model_shape : split_model_shape*2])(split_input)
            B = layers.Lambda(lambda x: x[:, split_model_shape*2 : split_model_shape*3])(split_input)
            A_out = split_model(layers.Subtract()([P, A]))
            B_out = split_model(layers.Subtract()([P, B]))
            split_out = layers.concatenate([A_out, B_out], axis=-1)
            merged_model = models.Model(inputs=split_input, outputs=split_out, name='split_model_{}'.format(i))
            all_models.append(merged_model)

        # One model 
        all_models.append(build_mlp_model(feat_shape, model_num=len(all_models), dense_layer_size=128, dropout_rate=0.8))

        lambd = 0.02 # L2 regularization
        # Combine all models into one model

        # Concatenation of 5 models outputs
        merged_out = layers.concatenate([model.output for model in all_models])
        merged_out = layers.Dense(3, name = 'merged_output', kernel_regularizer = regularizers.l2(lambd), kernel_initializer=initializers.glorot_uniform(seed=self.seed))(merged_out)
        merged_out = layers.BatchNormalization(name = 'merged_bn')(merged_out)
        merged_out = layers.Activation('softmax')(merged_out)

        # The final combined model.
        combined_model = models.Model([model.input for model in all_models], outputs = merged_out, name = 'merged_model')
        #print(combined_model.summary())

        return combined_model


    def train_model(self, embeddings, features, input_files):
        """
        Train the model, predict and write a submission.
        """

        # Load and assemble data from multiple files.
        test_ids = {}
        test_ids_list = []
        Y_test = []
        # embeddings - a list of embeddings files from two different BERT models:
        # 0: BERT Large Uncased 
        # 1: BERT Large Cased
        for model_i, embeddings_files in enumerate(embeddings):

            for name in ['test', 'train']:
                print('Processing {} datasets'.format(name))

                # Load the embeddings from the .json files
                for file_i, embeddings_file in enumerate(embeddings_files[name]):
                    if file_i == 0:
                        data, idx_map = self.load_embeddings(
                                embeddings_file,
                                data = {'emb': [], 'df_ids': []},
                                idx_map = {})
                    else:
                        data, idx_map = self.load_embeddings(embeddings_file, data, idx_map)

                # Concatenate three target embeddings into one NumPy array
                # For each sample we have an embeddings array of size 3*3*1024
                for i, emb in enumerate(data['emb']):
                    data['emb'][i] = np.concatenate(emb)

                # Load the features and the labels for each sample.
                if model_i == 0:
                    feats = [[] for _ in range(len(data['emb']))]
                    labels = [[0, 0, 0] for _ in range(len(data['emb']))]

                    if name == 'train':
                        X_emb_train = [np.array(data['emb'])]
                    else:
                        X_emb_test = [np.array(data['emb'])]

                    # Load features
                    for features_i, filename in enumerate(features[name]):
                        feats = self.load_feats(filename, features['{}_ids'.format(name)][features_i], feats, idx_map)

                    if name == 'train':
                        X_feats_train = np.array(feats)
                    else:
                        X_feats_test = np.array(feats)

                    # Load labels
                    for filename in input_files[name]:
                        labels = self.load_labels(filename, labels, idx_map)

                    if name == 'train':
                        Y_train = np.array(labels)
                    else:
                        for data_i, idx in enumerate(data['df_ids']):
                            if idx not in test_ids:
                                test_ids_list.append(idx)
                                test_ids[idx] = len(test_ids)
                                Y_test.append(labels[data_i])
                        Y_test = np.array(Y_test)
                else:
                    if name == 'train':
                        X_emb_train.append(np.array(data['emb']))
                    else:
                        X_emb_test.append(np.array(data['emb']))

        print('Train shape:', [x.shape for x in X_emb_train], X_feats_train.shape)
        print('Test shape:', [x.shape for x in X_emb_test], X_feats_train.shape)

        # Normalise feats
        all_feats = np.concatenate([X_feats_train, X_feats_test])
        all_max = np.max(all_feats, axis=0)
        X_feats_train /= all_max
        X_feats_test /= all_max

        model_shapes = [X_feats_train.shape[1]] + [x.shape[1] for x in X_emb_train]
        X_test = X_emb_test + X_emb_test + [X_feats_test]

        Y_test = np.array(Y_test)
        prediction = np.zeros((len(test_ids), 3)) # testing predictions
        prediction_cnt = np.zeros((len(test_ids), 3)) # testing predictions counts

        learning_rate = 0.02
        decay = 0.03
        n_fold = 5
        batch_size = 64
        epochs = 10000
        patience = 50

        # Training and cross-validation
        folds = KFold(n_splits=n_fold, shuffle=True, random_state=1122)

        scores = []
        for fold_n, (train_index, valid_index) in enumerate(folds.split(Y_train)):
            # split training and validation data
            print('Fold', fold_n, 'started at', time.ctime())
            X_tr = [x[train_index] for x in X_emb_train] + [x[train_index] for x in X_emb_train] + [X_feats_train[train_index]]
            X_val = [x[valid_index] for x in X_emb_train] + [x[valid_index] for x in X_emb_train] + [X_feats_train[valid_index]]
            Y_tr, Y_val = Y_train[train_index], Y_train[valid_index]

            # Define the model, re-initializing for each fold
            classif_model = self.get_model(model_shapes)
            classif_model.compile(optimizer = optimizers.Adam(lr = learning_rate, decay = decay), loss = "categorical_crossentropy")
            callbacks = [kc.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights = True)]

            # train the model
            classif_model.fit(x = X_tr, y = Y_tr, epochs = epochs, batch_size = batch_size, callbacks = callbacks, validation_data = (X_val, Y_val), verbose = 0)

            # make predictions on validation and test data
            pred_valid = classif_model.predict(x = X_val, verbose = 0)
            pred = classif_model.predict(x = X_test, verbose = 0)

            print('Stopped at {}, score {}'.format(callbacks[0].stopped_epoch, log_loss(Y_val, pred_valid)))

            scores.append(log_loss(Y_val, pred_valid))
            for i, idx in enumerate(test_ids_list):
                prediction[test_ids[idx]] += pred[i]
                prediction_cnt[test_ids[idx]] += np.ones(3)

        # Print CV scores, as well as score on the test data
        print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
        print(scores)
        print('Test score:', log_loss(Y_test, prediction/prediction_cnt))

        prediction /= prediction_cnt

        # Write the prediction to file for submission
        submission = pd.DataFrame.from_records([(idx, 0.0, 0.0, 0.0) for idx in test_ids_list], columns=['ID', 'A', 'B', 'NEITHER'])
        submission['A'] = prediction[:,1]
        submission['B'] = prediction[:,2]
        submission['NEITHER'] = prediction[:,0]
        submission.to_csv('submission.csv', index=False)

