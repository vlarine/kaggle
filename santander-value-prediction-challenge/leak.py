import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from tqdm import tqdm

from santander_common import *

def get_pred(data, feats, extra_feats, offset = 2, start=0, min_nonzero=1):
    f1 = feats[:(offset * -1)]
    f2 = feats[offset:]
    for ef in extra_feats:
        f1 += ef[start:(offset * -1)]
        f2 += ef[offset + start:]

    if min_nonzero > 0:
        d0 = data[f1]
        d0 = d0[(d0 > 0).sum(axis=1) < min_nonzero]
    d1 = data[f1].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    if min_nonzero > 0 and d0.shape[0] > 0:
        d1.loc[d0.index, 'key'] = 0
    d2 = data[f2].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d2['pred'] = data[feats[offset-2]]
    d2 = d2[d2['pred'] != 0] # Keep?
    d3 = d2[~d2.duplicated(['key'], keep=False)]
    d4 = d1[~d1.duplicated(['key'], keep=False)]
    d5 = d4.merge(d3, how='inner', on='key')

    d = d1.merge(d5, how='left', on='key')
    return d.pred.fillna(0)

def get_all_pred(data, feats, extra_feats, max_offset):
    target = pd.Series(index=data.index, data=np.zeros(data.shape[0]))
    targets = {x: pd.Series(index=data.index, data=np.zeros(data.shape[0])) for x in range(2, max_offset + 1)}
    for offset in tqdm(range(2, max_offset + 1)):
    #for offset in range(2, 3):
        #for start in range(0, max(1, 20 - offset)):
        for start in range(0, 1):
            pred = get_pred(data, feats, extra_feats, offset, start, min_nonzero = 3 if offset > 37 else 1)
            mask = (target == 0) & (pred != 0)
            target[mask] = pred[mask]
            mask = pred != 0
            targets[offset][mask] = pred[mask]
    for offset in tqdm(range(2, 20)):
    #for offset in range(2, 3):
        for start in range(0, 1):
            pred = get_pred(data, feats, [feats], offset, start, min_nonzero=3)
            mask = (target == 0) & (pred != 0)
            target[mask] = pred[mask]
            mask = pred != 0
            targets[offset][mask] = pred[mask]
    return target, targets


def rewrite_datasets(pred, preds):
    new_data = {}
    col_names = []
    for i in range(get_names_len()):
        col_names += get_names(i)

    sub = pd.read_csv('../input/sample_submission.csv')
    sub.target = 0
    have_data = pred != 0
    sub.loc[have_data, 'target'] = pred[have_data]
    found_target = sub.set_index('ID').T.to_dict()

    with open('../input/train.csv') as f:
        line = f.readline()
        names = line.strip().split(',')[2:]
        col_order = [names.index(x) for x in col_names]
        for line in tqdm(f):
            arr = line.strip().split(',')
            data = arr[2:]
            new_data[arr[0]] = {'target': arr[1], 'is_train': 1, 'is_fake': 0, 'data': data}

    with open('../input/test.csv') as f:
        line = f.readline()
        for line in tqdm(f):
            arr = line.strip().split(',')
            data = arr[1:]
            new_data[arr[0]] = {'target': found_target[arr[0]]['target'] if arr[0] in found_target else -1, 'is_train': 0, 'is_fake': 1 if is_fake_data(data) else 0, 'data': data}

    new_order = col_order + [i for i in range(len(names)) if i not in col_order]
    new_names = [names[i] for i in new_order]
    with open('data/data_full.csv', 'w') as wf:
        wf.write('ID,target,is_train,is_fake,{}\n'.format(','.join(new_names)))
        for k, v in tqdm(sorted(new_data.items(), key=lambda kv: kv[0])):
            wf.write('{},{},{},{},{}\n'.format(k, v['target'], v['is_train'], v['is_fake'], ','.join([v['data'][i] for i in new_order])))


def process_train():
    train = pd.read_csv('../input/train.csv')
    names_len = get_names_len()
    extra_feats = [get_names(i) for i in range(names_len)]
    feats = get_names()
    for max_offset in range(35, 39+1):
        pred_train, preds_train = get_all_pred(train, feats, extra_feats, max_offset)
        have_data = pred_train != 0
        true_pred = pred_train[have_data] == train.target[have_data]
        print('Max lag {}: Score = {} on {} out of {} training samples'.format(max_offset, true_pred.sum(), have_data.sum(), train.shape[0]))

def process_test():
    test = pd.read_csv('../input/test.csv')
    names_len = get_names_len()
    extra_feats = [get_names(i) for i in range(names_len)]
    feats = get_names()
    pred_test, preds_test = get_all_pred(test, feats, extra_feats, 39)
    have_data = pred_test != 0
    print('Have predictions for {} out of {} test samples'.format(have_data.sum(), test.shape[0]))
    #return
    rewrite_datasets(pred_test, preds_test)

def main():
    process_test()

if __name__ == '__main__':
    main()

