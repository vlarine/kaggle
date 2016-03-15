#!/usr/bin/env python3
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
from sklearn import preprocessing
import os

# List of different feaatures for different models selected according featues importance
flist = ['feature 136', 'feature 111', 'feature 135', 'feature 318', 'feature 162', 'feature 178', 'feature 161', 'feature 220', 'feature 153', 'feature 46', 'feature 370', 'feature 205', 'feature 167', 'feature 206', 'feature 42', 'feature 85', 'feature 284', 'feature 160', 'feature 154', 'feature 163', 'feature 229', 'feature 177', 'feature 188', 'feature 62', 'feature 310', 'feature 362', 'feature 360', 'feature 172', 'feature 222', 'feature 183', 'feature 221', 'feature 207', 'feature 45', 'feature 305', 'feature 285', 'feature 228', 'feature 308', 'feature 182', 'feature 217', 'feature 179', 'feature 198', 'feature 196', 'feature 289', 'feature 39', 'feature 74', 'feature 56', 'feature 187', 'feature 290', 'feature 375', 'feature 218', 'feature 230', 'feature 181', 'feature 51', 'feature 234', 'feature 368', 'feature 191', 'feature 309', 'feature 44', 'feature 283', 'feature 75', 'feature 157', 'feature 155', 'feature 314', 'feature 86', 'feature 87', 'feature 345', 'feature 195', 'feature 134', 'feature 70', 'feature 291', 'feature 197', 'feature 273', 'feature 209', 'feature 81', 'feature 223', 'feature 219', 'feature 235', 'feature 376', 'feature 301', 'feature 315', 'feature 306', 'feature 55', 'feature 171', 'feature 202', 'feature 73', 'feature 227', 'feature 233', 'feature 193', 'feature 313', 'feature 307', 'feature 71', 'feature 68', 'feature 201', 'feature 232', 'feature 80', 'feature 312', 'feature 170', 'feature 54', 'feature 203', 'feature 82']
elist = ['event_type 31', 'event_type 21', 'event_type 35', 'event_type 34', 'event_type 20', 'event_type 15', 'event_type 54', 'event_type 11']

flist2 = ['feature 345', 'feature 160', 'feature 86', 'feature 205', 'feature 204', 'feature 182', 'feature 134', 'feature 289', 'feature 56', 'feature 273', 'feature 87', 'feature 83', 'feature 375', 'feature 196', 'feature 290', 'feature 55', 'feature 376', 'feature 172', 'feature 81', 'feature 291', 'feature 209', 'feature 368', 'feature 171', 'feature 283', 'feature 155', 'feature 202', 'feature 68', 'feature 179', 'feature 54', 'feature 71', 'feature 201', 'feature 193', 'feature 80', 'feature 170', 'feature 82', 'feature 203']
elist2 = ['event_type 15', 'event_type 49', 'event_type 23', 'event_type 34', 'event_type 54', 'event_type 35', 'event_type 11']



def load_data():
    """Load the data and do feature ingineering
    """

    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    event_type = pd.read_csv('../input/event_type.csv')
    log_feature = pd.read_csv('../input/log_feature.csv')
    resource_type = pd.read_csv('../input/resource_type.csv')
    severity_type = pd.read_csv('../input/severity_type.csv')

    train_dict = train['id'].to_dict()
    train_dict = dict((v, k) for k, v in train_dict.items())
    test_dict = test['id'].to_dict()
    test_dict = dict((v, k) for k, v in test_dict.items())


    # Add number of location appearance in train/test set
    loc_counts = train.location.append(test.location).value_counts()
    for i,row in train.iterrows():
        train.set_value(i, 'loc_counts', loc_counts[row.location])
    for i,row in test.iterrows():
        test.set_value(i, 'loc_counts', loc_counts[row.location])

    # Order of item appearance in raw data files makes sense
    appearance = {x:0 for x in range(1, 1126+1)}
    for i, row in severity_type.iterrows():
        idx = row['id']
        if idx in train_dict:
            location = train.loc[train_dict[idx], 'location']
            appearance[int(location)] += 1
            train.set_value(train_dict[idx], 'appearance', appearance[int(location)])
        else:
            location = test.loc[test_dict[idx], 'location']
            appearance[int(location)] += 1
            test.set_value(test_dict[idx], 'appearance', appearance[int(location)])

    train = pd.merge(train, severity_type, on='id')
    test = pd.merge(test, severity_type, on='id')

    # Binarise resource type
    resource_type_dict = {}
    dict_val = 1
    for name in pd.unique(resource_type.resource_type.ravel()):
        resource_type_dict[name] = dict_val
        dict_val *= 2
    train['resource_type_bin'] = 0
    test['resource_type_bin'] = 0
    for i,row in resource_type.iterrows():
        if row['id'] in train_dict:
            train.set_value(train_dict[row['id']], 'resource_type_bin', int(train.ix[train_dict[row['id']],'resource_type_bin']) + resource_type_dict[row.resource_type])
            train.set_value(train_dict[row['id']], row.resource_type, 1)
        if row['id'] in test_dict:
            test.set_value(test_dict[row['id']], 'resource_type_bin', int(test.ix[test_dict[row['id']],'resource_type_bin']) + resource_type_dict[row.resource_type])
            test.set_value(test_dict[row['id']], row.resource_type, 1)


    # Split train and test data for two models
    train2 = train.copy()
    test2 = test.copy()

    # Binarise event data
    event_type_dict = {}
    dict_val = 1
    for name in pd.unique(event_type.event_type.ravel()):
        event_type_dict[name] = dict_val
        dict_val *= 2
    train['event_type'] = 0
    test['event_type'] = 0
    train2['event_type'] = 0
    test2['event_type'] = 0
    train['event_type_cnt'] = 0
    test['event_type_cnt'] = 0
    train2['event_type_cnt'] = 0
    test2['event_type_cnt'] = 0
    for i,row in event_type.iterrows():
        if row['id'] in train_dict:
            train.set_value(train_dict[row['id']], 'event_type_cnt', train.ix[train_dict[row['id']],'event_type_cnt'] + 1)
            train2.set_value(train_dict[row['id']], 'event_type_cnt', train2.ix[train_dict[row['id']],'event_type_cnt'] + 1)
            if row.event_type in elist:
                train.set_value(train_dict[row['id']], 'event_type', train.ix[train_dict[row['id']],'event_type'] + event_type_dict[row.event_type])
                train.set_value(train_dict[row['id']], row.event_type, 1)
            if row.event_type in elist2:
                train2.set_value(train_dict[row['id']], 'event_type', train2.ix[train_dict[row['id']],'event_type'] + event_type_dict[row.event_type])
                train2.set_value(train_dict[row['id']], row.event_type, 1)
        if row['id'] in test_dict:
            test.set_value(test_dict[row['id']], 'event_type_cnt', test.ix[test_dict[row['id']],'event_type_cnt'] + 1)
            test2.set_value(test_dict[row['id']], 'event_type_cnt', test2.ix[test_dict[row['id']],'event_type_cnt'] + 1)
            if row.event_type in elist:
                test.set_value(test_dict[row['id']], 'event_type', test.ix[test_dict[row['id']],'event_type'] + event_type_dict[row.event_type])
                test.set_value(test_dict[row['id']], row.event_type, 1)
            if row.event_type in elist2:
                test2.set_value(test_dict[row['id']], 'event_type', test2.ix[test_dict[row['id']],'event_type'] + event_type_dict[row.event_type])
                test2.set_value(test_dict[row['id']], row.event_type, 1)

    # Make some features for pairs of events
    for i in range(4):
        name1 = elist[-i-1]
        for j in range(i+1, 4):
            name2 = elist[-j-1]
            feature_name = 'pair event {}_{}'.format(name1.split(' ')[1], name2.split(' ')[1])
            train.set_value(0, feature_name, 0)
            test.set_value(0, feature_name, 0)
            train.loc[(test[name1] > 0) & (train[name2] > 0), feature_name] = 1
            test.loc[(test[name1] > 0) & (test[name2] > 0), feature_name] = 1

    for i in range(4):
        name1 = elist2[-i-1]
        for j in range(i+1, 4):
            name2 = elist2[-j-1]
            feature_name = 'pair event {}_{}'.format(name1.split(' ')[1], name2.split(' ')[1])
            train2.set_value(0, feature_name, 0)
            test2.set_value(0, feature_name, 0)
            train2.loc[(test2[name1] > 0) & (train2[name2] > 0), feature_name] = 1
            test2.loc[(test2[name1] > 0) & (test2[name2] > 0), feature_name] = 1


    # Binarise log features
    train['log_feature'] = 0
    test['log_feature'] = 0
    train2['log_feature'] = 0
    test2['log_feature'] = 0
    for i,row in log_feature.iterrows():
        if row['id'] in train_dict:
            train.set_value(train_dict[row['id']], 'log_feature', train.ix[train_dict[row['id']],'log_feature'] + 1)
            train2.set_value(train_dict[row['id']], 'log_feature', train2.ix[train_dict[row['id']],'log_feature'] + 1)
            if row.log_feature in flist or True:
                train.set_value(train_dict[row['id']], row.log_feature, row.volume)
            if row.log_feature in flist2 or True:
                train2.set_value(train_dict[row['id']], row.log_feature, row.volume)
        if row['id'] in test_dict:
            test.set_value(test_dict[row['id']], 'log_feature', test.ix[test_dict[row['id']],'log_feature'] + 1)
            test2.set_value(test_dict[row['id']], 'log_feature', test2.ix[test_dict[row['id']],'log_feature'] + 1)
            if row.log_feature in flist or True:
                test.set_value(test_dict[row['id']], row.log_feature, row.volume)
            if row.log_feature in flist2 or True:
                test2.set_value(test_dict[row['id']], row.log_feature, row.volume)

    train = train.fillna(0)
    test = test.fillna(0)
    train2 = train2.fillna(0)
    test2 = test2.fillna(0)

    # Encode categorical features
    for name in train.columns:
        if train[name].dtypes == object:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[name].values) + list(test[name].values))
            train[name] = lbl.transform(list(train[name].values))
            test[name] = lbl.transform(list(test[name].values))

    for name in train2.columns:
        if train2[name].dtypes == object:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train2[name].values) + list(test2[name].values))
            train2[name] = lbl.transform(list(train2[name].values))
            test2[name] = lbl.transform(list(test2[name].values))


    idx = test['id'].astype(int)
    labels = train.fault_severity.astype(int)

    test = test.drop(['id'],axis=1)
    train = train.drop(['id', 'fault_severity'],axis=1)
    test2 = test2.drop(['id'],axis=1)
    train2 = train2.drop(['id', 'fault_severity'],axis=1)

    train = np.array(train).astype(float)
    test = np.array(test).astype(float)
    train2 = np.array(train2).astype(float)
    test2 = np.array(test2).astype(float)

    return train, train2, labels, test, test2, idx
