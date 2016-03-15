#!/usr/bin/env python3.4
# -*- coding: utf8 -*-

import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing
from scipy import stats
import os

sub_dir = 'subs/'
test_id  = pd.read_csv("../input/test.csv", usecols=['id'])
base = 0.4215                                               # Base score to count weigths
subs = ['0.42089_0', '0.42110_0', '0.42117_0', '0.42054_0'] # Submissions to combine

def getPred():
    """Combine selected submissions with weights according Public LB scores.
    """
    cnt = len(testId['id'].values)
    pred = np.zeros([cnt, 3], dtype=float)
    weights = []

    for sub in subs:
        weights.append(base - float(sub.split('_')[0]))

    sum_weights = sum(weights)
    for i, sub in enumerate(subs):
        w = weights[i] / sum_weights
        curr_pred = pd.read_csv(sub_dir + sub, usecols=['predict_0', 'predict_1', 'predict_2'])
        curr_pred = np.array(curr_pred).astype(float)
        pred += curr_pred*w

    return pred

pred = getPred()
res_df = pd.DataFrame(pred, columns=['predict_0', 'predict_1', 'predict_2' ])
res_df['id'] = test_id['id']
res_df.to_csv('res.csv', index=False)
