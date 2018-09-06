import numpy as np
import pandas as pd
from tqdm import tqdm
import json

from santander_common import *


def get_data_keys(data, shifts, col_order, target_col_order, target_back=2, nonzero_min=2):
    data_keys = {}
    for start in shifts:
        target_keys = []
        t_target_keys = []
        target_nonzero_cnt = 0
        search_keys = []
        t_search_keys = []
        search_nonzero_cnt = 0
        target = round(float(data[target_col_order[start - target_back]]))
        prev_value = -1
        for i, c in enumerate(col_order):
            col_value = round(float(data[c]))
            t_col_value = round(float(data[target_col_order[i]]))
            if i >= start:
                target_keys.append(col_value)
                t_target_keys.append(t_col_value)
                if col_value > 0 and col_value != prev_value:
                    target_nonzero_cnt += 1
            if i < (len(col_order) - start):
                search_keys.append(col_value)
                t_search_keys.append(t_col_value)
                if col_value > 0 and col_value != prev_value:
                    search_nonzero_cnt += 1
            prev_value = col_value

        data_keys[start] = {
            'target': target if target_nonzero_cnt >= nonzero_min and target > 0 else -1,
            'target_nonzero_cnt': target_nonzero_cnt,
            'target_key': '_'.join([str(x) for x in target_keys]) if target_nonzero_cnt >= nonzero_min and target > 0 else '',
            't_target_key': '_'.join([str(x) for x in t_target_keys]),
            'search_key': '_'.join([str(x) for x in search_keys]) if search_nonzero_cnt >= nonzero_min else '',
            't_search_key': '_'.join([str(x) for x in t_search_keys]),
        }

    return data_keys

def has_int_value(data):
    for d in data:
        float_d = float(d)
        round_d = float(round(float_d))
        if float_d > 0 and float_d == round_d:
            #print(d)
            return True
    return False


def is_fake_data(data):
    for d in data:
        if len(d.split('.')[1]) > 2:
            return not has_int_value(data)
            #if has_int_value(data):
            #    print('HAS INT!!!')
            #    print(data)
            #    exit()
            return True
    return False


def process(found_target=None, findings = None, round_num=0, nonzero_min=2):
    col_names = get_names(round_num)
    #print(len(col_names))
    #print(col_names)
    #exit()
    target_names = get_names()
    shifts = [x for x in range(2, len(col_names) - 2 + 1 - 1)]
    targets_train = {x: {} for x in shifts}
    targets_test = {x: {} for x in shifts}
    save_zeros = False

    with open('../input/train.csv') as f:
        line = f.readline()
        names = line.strip().split(',')[2:]
        col_order = [names.index(x) for x in col_names]
        target_order = [names.index(x) for x in target_names]
        for line in tqdm(f):
            arr = line.strip().split(',')
            data = arr[2:]
            data_keys = get_data_keys(data, shifts, col_order, target_order, nonzero_min=nonzero_min)
            for start in shifts:
                if data_keys[start]['target'] > 0 or save_zeros:
                    target = data_keys[start]['target']
                    target_key = data_keys[start]['target_key']
                    t_target_key = data_keys[start]['t_target_key']
                    if target_key in targets_train[start]:
                        if target != targets_train[start][target_key]:
                            #print('Error')
                            #print(arr[0], target_key, target, targets[start][target_key])
                            pass
                    else:
                        targets_train[start][target_key] = (target, arr[0], data_keys[start]['target_nonzero_cnt'], start, t_target_key)

    #print('Found keys in train:', len(targets[2]))
    #return

    fake_data = set()
    key_index = {}
    with open('../input/test.csv') as f:
        line = f.readline()
        for line in tqdm(f):
            arr = line.strip().split(',')
            data = arr[1:]
            if is_fake_data(data):
                fake_data.add(arr[0])
                continue
            data_keys = get_data_keys(data, shifts, col_order, target_order, nonzero_min=nonzero_min)
            for start in shifts:
                if data_keys[start]['target'] > 0 or save_zeros:
                    target = data_keys[start]['target']
                    target_key = data_keys[start]['target_key']
                    t_target_key = data_keys[start]['t_target_key']
                    if start == 2:
                        key_index[target_key] = arr[0]
                    if target_key in targets_test[start]:
                        if target != targets_test[start][target_key][0]:
                            #print('Error')
                            #print(target_key, target, targets[start][target_key])
                            pass
                    else:
                        targets_test[start][target_key] = (target, arr[0], data_keys[start]['target_nonzero_cnt'], start, t_target_key)

    #print('Found keys in train and test:', len(targets[2]))
    #print('Num fake data:', len(fake_data))

    if found_target is None:
        found_target = {}

    if findings is None:
        findings = {}

    with open('../input/train.csv') as f:
        line = f.readline()
        for line in f:
            arr = line.strip().split(',')
            data = arr[2:]
            data_keys = get_data_keys(data, shifts, col_order, target_order, nonzero_min=nonzero_min)
            for start in shifts:
                search_key = data_keys[start]['search_key']
                t_search_key = data_keys[start]['t_search_key']
                if search_key != '' and search_key in targets_train[start]:
                    parent_id = targets_train[start][search_key][1]
                    child_id = arr[0]
                    if parent_id not in findings:
                        findings[parent_id] = {}
                    findings[parent_id][child_id] = start
                #if search_key != '' and search_key in targets[start] and targets[start][search_key][4] == t_search_key and \
                if search_key != '' and search_key in targets_train[start] and \
                    (arr[0] not in found_target or targets_train[start][search_key][2] > found_target[arr[0]][1] or (targets_train[start][search_key][2] == found_target[arr[0]][1] and targets_train[start][search_key][3] < found_target[arr[0]][3])):
                    found_target[arr[0]] = (targets_train[start][search_key][0], targets_train[start][search_key][2], round_num, targets_train[start][search_key][3])

    found_train_good = 0
    found_train_bad = 0
    with open('../input/train.csv') as f:
        line = f.readline()
        for line in f:
            arr = line.strip().split(',')
            data = arr[2:]
            if arr[0] in found_target:
                train_target = round(float(arr[1]))
                train_found_target = round(float(found_target[arr[0]][0]))
                if train_found_target == train_target:
                    found_train_good += 1
                else:
                    found_train_bad += 1
    if found_train_good + found_train_bad > 0:
        with open('default.log', 'a') as wf:
            wf.write('*** Train valid {} {} {}/{}\n'.format(found_train_good + found_train_bad, found_train_bad, found_train_good, found_train_good / (found_train_good + found_train_bad)))
        print('*** Train valid {} {}/{}'.format(found_train_bad, found_train_good, found_train_good / (found_train_good + found_train_bad)))

    #return found_target, fake_data
    #print('Found targets in train:', len(found_target))
    #found_target = {}

    with open('../input/test.csv') as f:
        line = f.readline()
        for line in tqdm(f):
            arr = line.strip().split(',')
            data = arr[1:]
            if is_fake_data(data):
                continue
            data_keys = get_data_keys(data, shifts, col_order, target_order, nonzero_min=nonzero_min)
            for start in shifts:
                search_key = data_keys[start]['search_key']
                t_search_key = data_keys[start]['t_search_key']
                if search_key != '' and search_key in targets_test[start]:
                    parent_id = targets_test[start][search_key][1]
                    child_id = arr[0]
                    if parent_id not in findings:
                        findings[parent_id] = {}
                    findings[parent_id][child_id] = start
                #if search_key != '' and search_key in targets[start] and targets[start][search_key][4] == t_search_key and \
                if search_key != '' and search_key in targets_test[start] and \
                    (arr[0] not in found_target or targets_test[start][search_key][2] > found_target[arr[0]][1] or (targets_test[start][search_key][2] == found_target[arr[0]][1] and targets_test[start][search_key][3] < found_target[arr[0]][3])):
                    found_target[arr[0]] = (targets_test[start][search_key][0], targets_test[start][search_key][2], round_num, targets_test[start][search_key][3])

    print('Found targets in train and test:', len(found_target))
    return found_target, fake_data, findings


def rewrite_datasets(found_target, fake_data):
    new_data = {}
    col_names = []
    for i in range(4):
        col_names += get_names(i)

    with open('../input/train.csv') as f:
        line = f.readline()
        names = line.strip().split(',')[2:]
        col_order = [names.index(x) for x in col_names]
        for line in f:
            arr = line.strip().split(',')
            data = arr[2:]
            new_data[arr[0]] = {'target': arr[1], 'found_target': found_target[arr[0]][0] if arr[0] in found_target else -1, 'found_len': found_target[arr[0]][1] if arr[0] in found_target else -1, 'round_num': found_target[arr[0]][2] if arr[0] in found_target else -1, 'is_train': 1, 'is_fake': 0, 'data': data}

    with open('../input/test.csv') as f:
        line = f.readline()
        for line in f:
            arr = line.strip().split(',')
            data = arr[1:]
            new_data[arr[0]] = {'target': -1, 'found_target': found_target[arr[0]][0] if arr[0] in found_target else -1, 'found_len': found_target[arr[0]][1] if arr[0] in found_target else -1, 'round_num': found_target[arr[0]][2] if arr[0] in found_target else -1, 'is_train': 0, 'is_fake': 1 if arr[0] in fake_data else 0, 'data': data}

    new_order = col_order + [i for i in range(len(names)) if i not in col_order]
    new_names = [names[i] for i in new_order]
    with open('data/data.csv', 'w') as wf:
        wf.write('ID,is_train,is_fake,target,found_target,found_len,round_num,{}\n'.format(','.join(new_names)))
        for k, v in tqdm(sorted(new_data.items(), key=lambda kv: kv[0])):
            wf.write('{},{},{},{},{},{},{},{}\n'.format(k, v['is_train'], v['is_fake'], v['target'], v['found_target'], v['found_len'], v['round_num'], ','.join([v['data'][i] for i in new_order])))


def run_leak():
    found_target = {}
    findings = {}
    for i in range(get_names_len()):
        print('*** Process group', i)
        found_target, fake_data, findings = process(found_target=found_target, findings=findings, round_num=i)


    rewrite_datasets(found_target, fake_data)


def main():
    run_leak()


if __name__ == '__main__':
    main()

