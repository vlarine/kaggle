#!/usr/bin/env python3
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics, grid_search
from sklearn.cross_validation import KFold

import utils
SEED = 1122 #13#42#1122


def get_score(train, labels, p):
    """Count CV score on 10 folds.
    """
    params = {'objective': 'binary:logistic',
              'eta': p.get('eta', 0.05),
              'max_depth': p.get('max_depth', 9),
              'min_child_weight': p.get('child', 1),
              'silent': 1,
              'subsample': p.get('sub', 0.85),
              'colsample_bytree': p.get('col', 0.6),
              'gamma': p.get('gamma', 0.7),
              'eval_metric': 'logloss',
              'seed': SEED}

    num_trees = p.get('num_trees', 500)
    n = 10

    score = 0
    kf = KFold(train.shape[0], n_folds=n, shuffle=True, random_state=SEED)

    for train_index, test_index in kf:
        xtrain = xgb.DMatrix(train[train_index], label=labels[train_index])
        xtest = xgb.DMatrix(train[test_index], label=labels[test_index])

        clr = xgb.train(params, xtrain, num_trees)
        pred = clr.predict(xtest)
        score += metrics.log_loss(labels[test_index], pred)

    return score / n

def get_params(train, labels, model=0):
    """Do a grid search for two models.
    """
    param_grid = [
        {'eta': [0.009], 'num_trees': [1500], 'max_depth': [13], 'child': [0.40], 'sub': [0.95], 'col': [0.65], 'gamma': [0.77]},
        {'eta': [0.009], 'num_trees': [1000], 'max_depth': [6], 'child': [0.57], 'sub': [0.75], 'col': [0.60], 'gamma': [0.93]},
    ][model]

    best_p = {}
    best_score = 1
    for p in grid_search.ParameterGrid(param_grid):
        cv_score, cv_scores, cp_preds = get_score(train, labels, p)
        if (cv_score < best_score):
            best_score = cv_score
            best_p = p

    print(best_p)
    return best_p

def train_and_pred(train, train2, labels, test, test2, columns, columns2):
    """Make a predictions using two models:
    First model: if there was a fault or not
    Secod model: if a falt was severe or not
    """
    labels = np.array(labels)

    labels1 = np.array(labels)
    labels1[labels > 0] = 1
    p = get_params(train, labels1)

    y = np.array(labels1).astype('int')
    X = train

    print('Train an XGBoost model1')
    params = {'objective': 'binary:logistic',
              'eta': p.get('eta', 0.05),
              'max_depth': p.get('max_depth', 9),
              'min_child_weight': p.get('child', 1),
              'silent': 1,
              'subsample': p.get('sub', 0.85),
              'colsample_bytree': p.get('col', 0.6),
              'gamma': p.get('gamma', 0.7),
              'eval_metric': 'logloss',
              'seed': SEED}
    num_trees = p.get('num_trees', 500)

    xtest = xgb.DMatrix(test)

    xtrain = xgb.DMatrix(X, label=y)
    clr = xgb.train(params, xtrain, num_trees)

    print('Predict from the XGBoost model1')
    pred1 = clr.predict(xtest)

    labels2 = np.array(labels[labels > 0])
    labels2[labels2 > 1] = 0
    p = get_params(train2[labels > 0], labels2, model=1)

    y = np.array(labels2).astype('int')
    X = train2[labels > 0]

    print('Train an XGBoost model2')
    params = {'objective': 'binary:logistic',
              'eta': p.get('eta', 0.05),
              'max_depth': p.get('max_depth', 9),
              'min_child_weight': p.get('child', 1),
              'silent': 1,
              'subsample': p.get('sub', 0.85),
              'colsample_bytree': p.get('col', 0.6),
              'gamma': p.get('gamma', 0.7),
              'eval_metric': 'logloss',
              'seed': SEED}

    num_trees = p.get('num_trees', 500)
    num_trees = p['num_trees'] if 'num_trees' in p else 500
    xtest = xgb.DMatrix(test2)

    xtrain = xgb.DMatrix(X, label=y)
    clr = xgb.train(params, xtrain, num_trees)

    print('Predict from the XGBoost model2')
    pred2 = clr.predict(xtest)

    pred = []
    for i, p1 in enumerate(pred1):
        v1 = 1.0 - p1
        v2 = p1 * pred2[i]
        v3 = 1.0 - v1 - v2
        pred.append([v1, v2, v3])

    return np.array(pred)

def main():
    train, train2, labels, test, test2, idx = utils_bin.load_data()

    pred = train_and_pred(train, train2, labels, test, test2)

    res_df = pd.DataFrame(pred, columns=['predict_0', 'predict_1', 'predict_2' ])
    res_df['id'] = idx
    res_df.to_csv('res.csv', index=False)

if __name__ == '__main__':
    main()
