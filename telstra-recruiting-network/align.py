#!/usr/bin/env python3.4
# -*- coding: utf8 -*-

import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing
from scipy import stats
import os

sub_dir = 'subs/'
test_id  = pd.read_csv("../input/test.csv", usecols=['id'])

def getPred():
    """ Align a submission to the train probability distribution
    """
    label_rate = [0.6481506570925348, 0.2534886871697602, 0.09836065573770492]

    filename = 'subs/0.42050_0'
    pred = pd.read_csv(filename, usecols=['predict_0', 'predict_1', 'predict_2'])
    pred = np.array(pred).astype(float)

    print(pred[:10])
    pred_rate = np.sum(pred, axis=0) / np.sum(pred)
    print(pred_rate)
    pred[:,0] *= label_rate[0]/pred_rate[0]
    pred[:,1] *= label_rate[1]/pred_rate[1]
    pred[:,2] *= label_rate[2]/pred_rate[2]
    print(pred[:10])
    pred_rate = np.sum(pred, axis=0) / np.sum(pred)
    print(pred_rate)
    return pred


pred = getPred()
res_df = pd.DataFrame(pred, columns=['predict_0', 'predict_1', 'predict_2' ])
res_df['id'] = test_id['id']
res_df.to_csv('res.csv', index=False)
