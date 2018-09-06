import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
from sklearn import model_selection
import eli5
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV
import json

from santander_common import *
from const import *

import math

SEED = 1122

#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5


def preprocess_leak_col():
    df = pd.read_csv('data/data_full.csv')
    train = df[df.is_train == 1]
    drop_col = 'ID,is_train,is_fake,target,found_target,found_len,round_num'.split(',')
    col = [c for c in train.columns if c not in drop_col]
    leak_col = []
    for c in col:
        leak1 = np.sum((train[c]==train['target']).astype(int))
        leak2 = np.sum((((train[c] - train['target']) / train['target']) < 0.05).astype(int))
        if leak1 > 30 and leak2 > 3500:
            leak_col.append(c)
    print(len(leak_col))

    col = list(leak_col)

#https://www.kaggle.com/johnfarrell/baseline-with-lag-select-fake-rows-dropped
    df["leak_nz_mean"] = df[col].apply(lambda x: x[x!=0].mean(), axis=1)
    df["leak_nz_max"] = df[col].apply(lambda x: x[x!=0].max(), axis=1)
    df["leak_nz_min"] = df[col].apply(lambda x: x[x!=0].min(), axis=1)
    df["leak_ez"] = df[col].apply(lambda x: len(x[x==0]), axis=1)
    df["leak_mean"] = df[col].apply(lambda x: x.mean(), axis=1)
    df["leak_max"] = df[col].apply(lambda x: x.max(), axis=1)
    df["leak_min"] = df[col].apply(lambda x: x.min(), axis=1)

    col = list(set(col) - set(get_names())) + ['leak_nz_mean', 'leak_nz_max', 'leak_nz_min', 'leak_ez', 'leak_mean', 'leak_max', 'leak_min']
    print(col)
    df[['ID', ] + col].to_csv('data/feat_leak.csv', index=False)


def preprocess_counts0():
    df = pd.read_csv('data/data_44.csv')
    col = get_names()
    df["nz_mean"] = df[col].apply(lambda x: x[x!=0].mean(), axis=1)
    df["nz_max"] = df[col].apply(lambda x: x[x!=0].max(), axis=1)
    df["nz_min"] = df[col].apply(lambda x: x[x!=0].min(), axis=1)
    df["ez"] = df[col].apply(lambda x: len(x[x==0]), axis=1)
    df["mean"] = df[col].apply(lambda x: x.mean(), axis=1)
    df["max"] = df[col].apply(lambda x: x.max(), axis=1)
    df["min"] = df[col].apply(lambda x: x.min(), axis=1)
    save_cols = ['nz_mean', 'nz_max', 'nz_min', 'ez', 'mean', 'max', 'min']
    df[save_cols + ['ID', ]].to_csv('data/feat_counts0.csv', index=False)


def preprocess_counts_groups():
    df = pd.read_csv('data/data_full.csv')
    col = get_names()
    save_cols = []
    for i in tqdm(range(1, get_names_len())):
        col = get_names(i)
        df['gr{}_nz_mean'.format(i)] = df[col].apply(lambda x: x[x!=0].mean(), axis=1)
        df['gr{}_ez'.format(i)] = df[col].apply(lambda x: len(x[x==0]), axis=1)
        save_cols += ['gr{}_nz_mean'.format(i), 'gr{}_ez'.format(i)]
    df[save_cols + ['ID', ]].to_csv('data/feat_counts_groups.csv', index=False)


def preprocess_counts_days():
    df = pd.read_csv('data/data_full.csv')
    col = get_names()
    save_cols = []
    names = [get_names(i) for i in range(get_names_len())]
    for i in tqdm(range(40)):
        col = [n[i] for n in names]
        col = get_names(i)
        df['day{}_nz_mean'.format(i)] = df[col].apply(lambda x: x[x!=0].mean(), axis=1)
        df['day{}_ez'.format(i)] = df[col].apply(lambda x: len(x[x==0]), axis=1)
        save_cols += ['day{}_nz_mean'.format(i), 'day{}_ez'.format(i)]
    df[save_cols + ['ID', ]].to_csv('data/feat_counts_days.csv', index=False)


def preprocess_numbers():
    with open('data/data_full.csv') as f, open('data/feat_numbers.csv', 'w') as wf:
        f.readline()
        col = ['ID',] + ['cnt_nz', 'len_nz', 'cnt_mode_nz', 'cnt_div', 'cnt_fake', 'cnt_norm'] \
            + ['cnt_d{}{}'.format(y, x) for x in range(10) for y in range(7)]
        wf.write(','.join(col) + '\n')
        for line in tqdm(f):
            arr = line.strip().split(',')
            data = arr[4:]
            cnt_nz = 0
            nz_values = {}
            cnt_div = 0
            cnt_fake = 0
            cnt_weird = 0
            cnt_dig = [[0 for x in range(10)] for y in range(7)]
            for d in data:
                if is_nonzero(d):
                    cnt_nz += 1
                    nz_values[d] = nz_values.get(d, 0) + 1
                if is_divided_number(d) or is_weird_number(d):
                    cnt_div += 1
                if is_fake_number(d):
                    cnt_fake += 1
                part = d.split('.')
                if len(part[0]) > 3 and part[0][-3:] == '000':
                    for pos, digit in enumerate(part[0][:-3][::-1]):
                        cnt_dig[pos][int(digit)] += 1

            val = [arr[0], ] + [str(cnt_nz), str(len(nz_values)), str(max(nz_values.values())), \
                   str(cnt_div), str(cnt_fake), str(cnt_nz - cnt_fake)] \
                + [str(cnt_dig[y][x]) for x in range(10) for y in range(7)]
            wf.write(','.join(val) + '\n')

def preprocess_numbers_core(n):
    core = set(get_names(n))
    print(core)
    with open('data/data_full.csv') as f, open('data/feat_numbers{}.csv'.format(n), 'w') as wf:
        names = f.readline().strip().split(',')[4:]
        col = ['ID',] + ['cnt_nz{}'.format(n), 'len_nz{}'.format(n), 'cnt_mode_nz{}'.format(n), 'cnt_div{}'.format(n), 'cnt_fake{}'.format(n), 'cnt_norm{}'.format(n)] \
            + ['cnt_d{}{}{}'.format(n, y, x) for x in range(10) for y in range(7)]
        wf.write(','.join(col) + '\n')
        for line in tqdm(f):
            arr = line.strip().split(',')
            data = arr[4:]
            cnt_nz = 0
            nz_values = {}
            cnt_div = 0
            cnt_fake = 0
            cnt_weird = 0
            cnt_dig = [[0 for x in range(10)] for y in range(7)]
            for i, d in enumerate(data):
                if names[i] not in core:
                    continue
                if is_nonzero(d):
                    cnt_nz += 1
                    nz_values[d] = nz_values.get(d, 0) + 1
                if is_divided_number(d) or is_weird_number(d):
                    cnt_div += 1
                if is_fake_number(d):
                    cnt_fake += 1
                part = d.split('.')
                if len(part[0]) > 3 and part[0][-3:] == '000':
                    for pos, digit in enumerate(part[0][:-3][::-1]):
                        cnt_dig[pos][int(digit)] += 1

            val = [arr[0], ] + [str(cnt_nz), str(len(nz_values)), str(max(nz_values.values())) if len(nz_values) > 0 else '0', \
                   str(cnt_div), str(cnt_fake), str(cnt_nz - cnt_fake)] \
                + [str(cnt_dig[y][x]) for x in range(10) for y in range(7)]
            wf.write(','.join(val) + '\n')

def preprocess_digits0():
    core = get_names()
    with open('data/data_44.csv') as f, open('data/feat_digits0.csv', 'w') as wf:
        f.readline()
        wf.write('ID,dig0_mode,dig0_mean,dig0_nz\n')
        for line in tqdm(f):
            arr = line.strip().split(',')
            data = arr[4:]
            cnt_dig = [[0 for x in range(10)] for y in range(7)]
            for i, d in enumerate(data):
                part = d.split('.')
                if len(part[0]) > 3 and part[0][-3:] == '000':
                    for pos, digit in enumerate(part[0][:-3][::-1]):
                        cnt_dig[pos][int(digit)] += 1
            factor = 1000
            mode_number = 0
            for d in np.array(cnt_dig).argmax(axis=1):
                mode_number += factor * d
                factor *= 10
            factor = 1000
            mean_number = 0
            for d in np.array(cnt_dig).mean(axis=1):
                mean_number += factor * d
                factor *= 10
            factor = 1000
            nz_number = 0
            for d in np.count_nonzero(np.array(cnt_dig), axis=1):
                nz_number += factor * d
                factor *= 10
            factor = 1000
            wf.write('{},{},{},{}\n'.format(arr[0], mode_number, mean_number, nz_number))


def preprocess_digits(n):
    core = get_names(n)
    with open('data/data_44.csv') as f, open('data/feat_digits{}.csv'.format(n), 'w') as wf:
        f.readline()
        wf.write('ID,dig{}_mode,dig{}_mean,dig{}_nz\n'.format(n,n,n))
        for line in tqdm(f):
            arr = line.strip().split(',')
            data = arr[4:]
            cnt_dig = [[0 for x in range(10)] for y in range(7)]
            for i, d in enumerate(data):
                part = d.split('.')
                if len(part[0]) > 3 and part[0][-3:] == '000':
                    for pos, digit in enumerate(part[0][:-3][::-1]):
                        cnt_dig[pos][int(digit)] += 1
            factor = 1000
            mode_number = 0
            for d in np.array(cnt_dig).argmax(axis=1):
                mode_number += factor * d
                factor *= 10
            factor = 1000
            mean_number = 0
            for d in np.array(cnt_dig).mean(axis=1):
                mean_number += factor * d
                factor *= 10
            factor = 1000
            nz_number = 0
            for d in np.count_nonzero(np.array(cnt_dig), axis=1):
                nz_number += factor * d
                factor *= 10
            factor = 1000
            wf.write('{},{},{},{}\n'.format(arr[0], mode_number, mean_number, nz_number))


def preprocess_full_stat():

    df = pd.read_csv('data/data_full.csv')
    df_id = df.ID.values
    df.drop('ID,is_train,is_fake,target'.split(','), axis=1, inplace=True)
    df['full_sum'] = df.sum(axis=1)
    df['full_var'] = df.var(axis=1)
    df['full_nzvar'] = df.apply(lambda x: x[x!=0].var(), axis=1)
    df['full_median'] = df.median(axis=1)
    df['full_mean'] = df.mean(axis=1)
    df['full_nzmean'] = df.apply(lambda x: x[x!=0].mean(), axis=1)
    df['full_std'] = df.std(axis=1)
    df['full_nzstd'] = df.apply(lambda x: x[x!=0].std(), axis=1)
    df['ID'] = df_id
    save_col = ['full_sum', 'full_var', 'full_nzvar', 'full_median', 'full_mean', 'full_nzmean', 'full_std', 'full_nzstd']
    df[save_col + ['ID', ]].to_csv('data/feat_full_stat.csv', index=False)


def preprocess_repare():
    feat_factors = [3, 7, 9, 11, 13, 15, 17, 19, 31]
    rep_stat = {factor:0 for factor in get_all_repare_factors()}
    with open('data/data_full.csv') as f, open('data/data_repare.csv', 'w') as wf, open('data/feat_rep_factor.csv', 'w') as wf2:
        wf.write(f.readline())
        wf2.write(','.join(['ID', ] + ['rep_factor_' + str(x) for x in feat_factors]) + '\n')
        for line in tqdm(f):
            arr = line.strip().split(',')
            data = arr[4:]
            rep_data = []
            item_stat = {x:0 for x in feat_factors}
            for d in data:
                rep_d = repare_value(d)
                rep_data.append(rep_d[0])
                if rep_d[1] in rep_stat:
                    rep_stat[rep_d[1]] += 1
                if rep_d[1] in feat_factors:
                    item_stat[rep_d[1]] += 1
            wf.write(','.join(arr[:4] + rep_data) + '\n')
            wf2.write(','.join([arr[0]] + [str(item_stat[x]) for x in feat_factors]) + '\n')
    print(rep_stat)

def preprocess_rep_stat():
    df = pd.read_csv('data/data_repare.csv')
    save_cols = []
    for i in [0, 15, 12, 17, 1, 7]:
        col = get_names(i)
        df['rep_gr{}_nz_mean'.format(i)] = df[col].apply(lambda x: x[x!=0].mean(), axis=1)
        save_cols += ['rep_gr{}_nz_mean'.format(i)]
    df[save_cols + ['ID', ]].to_csv('data/feat_rep_stat.csv', index=False)


def preprocess_order():
    df = pd.read_csv('data/data_full.csv')
    save_cols = []
    for i in tqdm([0, 15, 12, 17, 1, 7]):
        col = get_names(i)
        def get_order(x):
            out = {'order_gr{}_o{}'.format(i, j):v for j, v in enumerate(sorted(list(x[col].values)))}
            return pd.Series(out)
        new_cols = ['order_gr{}_o{}'.format(i, j) for j in range(40)]
        df.append(new_cols)
        df[new_cols] = df.apply(get_order, axis=1)
        save_cols += new_cols
    df[save_cols + ['ID', ]].to_csv('data/feat_order.csv', index=False)
    #exit()


def preprocess_log():
    df = pd.read_csv('data/data_full.csv')
    save_cols = []
    for i in tqdm([0, 15, 12, 17, 1, 7]):
        col = get_names(i)
        df['gr{}_nz_mean_log'.format(i)] = df[col].apply(lambda x: np.log1p(x[x!=0].mean()), axis=1)
        df['gr{}_nz_log_mean'.format(i)] = df[col].apply(lambda x: np.log1p(x[x!=0]).mean(), axis=1)
        df['gr{}_nz_median_log'.format(i)] = df[col].apply(lambda x: np.log1p(x[x!=0].median()), axis=1)
        save_cols += ['gr{}_nz_mean_log'.format(i), 'gr{}_nz_log_mean'.format(i), 'gr{}_nz_median_log'.format(i)]
    df[save_cols + ['ID', ]].to_csv('data/feat_log.csv', index=False)


def preprocess_values():
    values = [20000000, 10000000, 2000000, 4000000, 1000000, 200000, 400000, 6000000, 600000, 40000000, 3000000]
    with open('data/data.csv') as f, open('data/feat_values.csv', 'w') as wf:
        f.readline()
        wf.write('ID,{}\n'.format(','.join([str(v) + '_cnt'  for v in values])))
        for line in tqdm(f):
            arr = line.strip().split(',')
            data = arr[4:]
            val_cnt = {str(v):0 for v in values}
            for d in data:
                d = d.split('.')[0]
                if d in val_cnt:
                    val_cnt[d] += 1
            wf.write('{},{}\n'.format(arr[0], ','.join([str(val_cnt[str(v)]) for v in values])))

def preprocess_str_values():
    def count_zeros(s):
        cnt = 0
        for c in s[::-1]:
            if c == '0':
                cnt += 1
            else:
                return cnt
        return cnt

    with open('data/data.csv') as f, open('data/feat_str_values.csv', 'w') as wf:
        f.readline()
        wf.write('ID,str_zeros_mean,str_len_mean,spec_mean,str_nz_mean,str_nz_mean_full\n')
        for line in tqdm(f):
            arr = line.strip().split(',')
            data = arr[4:]
            zeros_n = 0
            zeros_sum = 0
            nonzeros_summ = 0
            nonzeros_cnt = 0
            len_summ = 0
            spec_summ = 0
            for d in data:
                d = d.split('.')[0]
                len_summ += len(d)
                cnt = count_zeros(d)
                if cnt >= 3:
                    zeros_n += 1
                    zeros_sum += cnt
                    spec_summ += int(d)
                    for c in d:
                        if c != '0':
                            nonzeros_summ += int(c)
                            nonzeros_cnt += 1

            wf.write('{},{},{},{},{},{}\n'.format(arr[0], zeros_sum / zeros_n if zeros_n > 0 else 0, len_summ / len(data), spec_summ / zeros_n if zeros_n > 0 else 0, nonzeros_summ / nonzeros_cnt if nonzeros_cnt > 0 else 0, nonzeros_summ / zeros_n if zeros_n > 0 else 0))

def preprocess_str_values0():
    def count_zeros(s):
        cnt = 0
        for c in s[::-1]:
            if c == '0':
                cnt += 1
            else:
                return cnt
        return cnt

    with open('data/data_44.csv') as f, open('data/feat_str_values0.csv', 'w') as wf:
        f.readline()
        wf.write('ID,str_zeros_mean0,str_len_mean0,spec_mean0,str_nz_mean0,str_nz_mean_full0\n')
        for line in tqdm(f):
            arr = line.strip().split(',')
            data = arr[4:]
            zeros_n = 0
            zeros_sum = 0
            nonzeros_summ = 0
            nonzeros_cnt = 0
            len_summ = 0
            spec_summ = 0
            for d in data:
                d = d.split('.')[0]
                len_summ += len(d)
                cnt = count_zeros(d)
                if cnt >= 3:
                    zeros_n += 1
                    zeros_sum += cnt
                    spec_summ += int(d)
                    for c in d:
                        if c != '0':
                            nonzeros_summ += int(c)
                            nonzeros_cnt += 1

            wf.write('{},{},{},{},{},{}\n'.format(arr[0], zeros_sum / zeros_n if zeros_n > 0 else 0, len_summ / len(data), spec_summ / zeros_n if zeros_n > 0 else 0, nonzeros_summ / nonzeros_cnt if nonzeros_cnt > 0 else 0, nonzeros_summ / zeros_n if zeros_n > 0 else 0))


def preprocess_core_sum_factor():
    cores_idx = [0, 1, 3, 15, 23, 25, 28] + [38, 45, 52, 54, 65, 66, 79] + [7, 11, 19, 31, 39, 49, 63] + [13, 2, 4, 14, 33, 41, 48] + [10, 5, 22, 36, 42, 53]
    core_names = [get_names(i) for i in cores_idx[:7]]
    with open('data/data_full.csv') as f, open('data/feat_core_sum_factor.csv', 'w') as wf:
        names = {n:i for i, n in enumerate(f.readline().strip().split(',')[4:])}
        wf.write('ID,{}\n'.format(','.join(['core_f{}d{}'.format(i, d) for d in range(40) for i in [0, 3]])))
        for line in tqdm(f):
            arr = line.strip().split(',')
            data = arr[4:]
            sum_factor_days = []
            for d in range(40):
                d_value = sum([float(data[names[core_names[i][d]]]) for i in range(1,7)])
                if d_value == 0:
                    for i in [0, 3]:
                        sum_factor_days.append(-1)
                else:
                    for i in [0, 3]:
                        sum_factor_days.append(float(data[names[core_names[i][d]]]) / d_value)
            wf.write('{},{}\n'.format(arr[0], ','.join([str(d) for d in sum_factor_days])))


def preprocess_simple_stat():
    df = pd.read_csv('data/data_full.csv')
    save_cols = []
    #for i in [0, ]:
    for i in [0, 15, 12, 17, 1, 7]:
        col = get_names(i)
        for j, row in tqdm(df.iterrows()):
            prev_value = 0
            prev2_value = 0
            prev3_value = 0
            time_delta = -1
            time2_delta = -1
            time3_delta = -1
            for d, value in enumerate(row[col].values):
                if value > 0:
                    if prev_value == 0:
                        prev_value = np.log1p(value)
                        time_delta = d
                    elif prev2_value == 0:
                        prev2_value = np.log1p(value)
                        time2_delta = d - time_delta - 1
                    else:
                        prev3_value = np.log1p(value)
                        time3_delta = d - time_delta - time2_delta - 2
                        break
            df.loc[j, 'c{}_prev_value'.format(i)] = prev_value
            df.loc[j, 'c{}_time_delta'.format(i)] = time_delta
            df.loc[j, 'c{}_prev2_value'.format(i)] = prev2_value
            df.loc[j, 'c{}_prev3_value'.format(i)] = prev3_value
            df.loc[j, 'c{}_time2_delta'.format(i)] = time2_delta
            df.loc[j, 'c{}_time3_delta'.format(i)] = time3_delta
        save_cols.append('c{}_prev_value'.format(i))
        save_cols.append('c{}_time_delta'.format(i))
        save_cols.append('c{}_prev2_value'.format(i))
        save_cols.append('c{}_prev3_value'.format(i))
        save_cols.append('c{}_time2_delta'.format(i))
        save_cols.append('c{}_time3_delta'.format(i))

    df[save_cols + ['ID', ]].to_csv('data/feat_simple_stat.csv', index=False)
    #exit()


def preprocess_pca():
    df = pd.read_csv('data/data_full.csv')
    df = df[df.is_fake==0]
    df_id = df.ID.values
    df.drop('ID,is_train,is_fake,target'.split(','), axis=1, inplace=True)
    save_col = []
    data = df.values
    for n_components in [2, 3]:
        pca = PCA(n_components=n_components, random_state=SEED)
        pca_X = pca.fit_transform(data)
        for i in range(n_components):
            df['pca{}_{}'.format(n_components, i)] = pca_X[:,i]
            save_col.append('pca{}_{}'.format(n_components, i))
    df['ID'] = df_id
    df[save_col + ['ID', ]].to_csv('data/feat_pca.csv', index=False)


def preprocess_pca_cores():
    df = pd.read_csv('data/data_full.csv')
    df = df[df.is_fake==0]
    df_id = df.ID.values
    df.drop('ID,is_train,is_fake,target'.split(','), axis=1, inplace=True)
    save_col = []
    for c in [0, 15, 12, 17, 1, 7]:
        col = get_names(c)
        data = df[col].values
        for n_components in [2, 3]:
            pca = PCA(n_components=n_components, random_state=SEED)
            pca_X = pca.fit_transform(data)
            for i in range(n_components):
                df['core{}_pca{}_{}'.format(c, n_components, i)] = pca_X[:,i]
                save_col.append('core{}_pca{}_{}'.format(c, n_components, i))
    df['ID'] = df_id
    df[save_col + ['ID', ]].to_csv('data/feat_pca_cores.csv', index=False)


def preprocess_simple_predict():
    df = pd.read_csv('data/data_full.csv')
    df = df[df.is_fake==0]
    res_df = df.ID.values
    df_target = df[df.target > 0].drop('ID,is_train,is_fake'.split(','), axis=1)
    target = df_target.target.values
    data = df_target.drop(['target',], axis=1).values.astype(int)
    val_sum = {}
    for i, dat in tqdm(enumerate(data)):
        for d in dat:
            if d <= 0:
                continue
            if d not in val_sum:
                val_sum[d] = [0, 0]
            val_sum[d][0] += target[i]
            val_sum[d][1] += 1
    df['simple_predict'] = 0
    for i, row in tqdm(df.drop('ID,is_train,is_fake,target'.split(','), axis=1).iterrows()):
        summ = 0
        cnt = 0.000001
        for val in row:
            if val not in val_sum or val_sum[val][1] < 10:
                continue
            summ += val_sum[val][0]
            cnt += val_sum[val][1]
        df.loc[i, 'simple_predict'] = summ / cnt
    df[['ID', 'simple_predict']].to_csv('data/feat_simple_predict.csv', index=False)


def preprocess():
    preprocess_counts0()
    preprocess_numbers()
    preprocess_full_stat()
    preprocess_counts_groups()
    preprocess_counts_days()
    preprocess_digits0()
    preprocess_digits(12)
    preprocess_digits(15)
    preprocess_repare()
    preprocess_rep_stat()
    preprocess_order()
    preprocess_log()
    preprocess_numbers_core(0)
    preprocess_values()
    preprocess_str_values()
    preprocess_str_values0()
    preprocess_core_sum_factor()
    preprocess_simple_stat()
    preprocess_pca()
    preprocess_pca_cores()
    preprocess_simple_predict()
    pass


def load_data():
    df = pd.read_csv('data/data_44.csv')
    df.fillna(0, inplace=True)
    col = get_names()


    df_add = pd.read_csv('data/feat_counts0.csv')
    df = pd.merge(df_add, df, on='ID', left_index=True)
    col += ['nz_mean', 'nz_max', 'nz_min', 'ez', 'mean', 'max']

    df_add = pd.read_csv('data/feat_numbers.csv')
    df = pd.merge(df_add, df, on='ID', left_index=True)
    col += ['cnt_nz', 'len_nz', 'cnt_mode_nz', 'cnt_div', 'cnt_norm'] \
         + ['cnt_d{}{}'.format(y, x) for x in range(10) for y in range(7)]

    df_add = pd.read_csv('data/feat_numbers0.csv')
    df = pd.merge(df_add, df, on='ID', left_index=True)
    col += ['len_nz0', 'cnt_norm0', ] \
         + ['cnt_d0{}{}'.format(y, x) for x in range(10) for y in range(7)]

    df_add = pd.read_csv('data/feat_full_stat.csv')
    df = pd.merge(df_add, df, on='ID', left_index=True)
    col += ['full_sum', 'full_var', 'full_nzvar', 'full_mean', 'full_nzmean', 'full_nzstd']

    df_add = pd.read_csv('data/feat_counts_groups.csv')
    df = pd.merge(df_add, df, on='ID', left_index=True)
    for i in range(1, get_names_len()):
        col += ['gr{}_nz_mean'.format(i), 'gr{}_ez'.format(i)]

    df_add = pd.read_csv('data/feat_counts_days.csv')
    df = pd.merge(df_add, df, on='ID', left_index=True)
    for i in range(40):
        col += ['day{}_nz_mean'.format(i), 'day{}_ez'.format(i)]

    df_add = pd.read_csv('data/feat_digits0.csv')
    df = pd.merge(df_add, df, on='ID', left_index=True)
    col += ['dig0_mode', 'dig0_mean', 'dig0_nz']

    df_add = pd.read_csv('data/feat_digits12.csv')
    df = pd.merge(df_add, df, on='ID', left_index=True)
    col += ['dig12_mode', 'dig12_mean', 'dig12_nz']

    df_add = pd.read_csv('data/feat_digits15.csv')
    df = pd.merge(df_add, df, on='ID', left_index=True)
    col += ['dig15_mode', 'dig15_mean', 'dig15_nz']

    df_add = pd.read_csv('data/feat_rep_factor.csv')
    df = pd.merge(df_add, df, on='ID', left_index=True)
    col_add = list(df_add.columns)
    col_add.remove('ID')
    col += col_add

    df_add = pd.read_csv('data/feat_rep_stat.csv')
    df = pd.merge(df_add, df, on='ID', left_index=True)
    col_add = list(df_add.columns)
    col_add.remove('ID')
    col += col_add

    df_add = pd.read_csv('data/feat_order.csv')
    df = pd.merge(df_add, df, on='ID', left_index=True)
    col_add = list(df_add.columns)
    col_add.remove('ID')
    col += col_add

    df_add = pd.read_csv('data/feat_log.csv')
    df = pd.merge(df_add, df, on='ID', left_index=True)
    col_add = list(df_add.columns)
    col_add.remove('ID')
    col += col_add

    df_add = pd.read_csv('data/feat_values.csv')
    df = pd.merge(df_add, df, on='ID', left_index=True)
    col_add = list(df_add.columns)
    col_add.remove('ID')
    col += col_add

    df_add = pd.read_csv('data/feat_str_values.csv')
    df = pd.merge(df_add, df, on='ID', left_index=True)
    col_add = list(df_add.columns)
    col_add.remove('ID')
    col += col_add

    df_add = pd.read_csv('data/feat_str_values0.csv')
    df = pd.merge(df_add, df, on='ID', left_index=True)
    col_add = list(df_add.columns)
    col_add.remove('ID')
    col += col_add

    print(df.shape)
    df_add = pd.read_csv('data/feat_ts_model_15_34.csv')
    df = pd.merge(df, df_add, on='ID', left_index=True, how='outer')
    col_add = list(df_add.columns)
    col_add.remove('ID')
    col += col_add

    df_add = pd.read_csv('data/feat_ts_model_20_34.csv')
    df = pd.merge(df, df_add, on='ID', left_index=True, how='outer')
    col_add = list(df_add.columns)
    col_add.remove('ID')
    col += col_add

    df_add = pd.read_csv('data/feat_ts_model_20_11.csv')
    df = pd.merge(df, df_add, on='ID', left_index=True, how='outer')
    col_add = list(df_add.columns)
    col_add.remove('ID')
    col += col_add
    print(df.shape)

    df_add = pd.read_csv('data/feat_core_sum_factor.csv')
    df = pd.merge(df, df_add, on='ID', left_index=True, how='outer')
    col_add = list(df_add.columns)
    col_add.remove('ID')
    col += col_add
    print(df.shape)

    df_add = pd.read_csv('data/feat_simple_stat.csv')
    df = pd.merge(df, df_add, on='ID', left_index=True, how='outer')
    col_add = list(df_add.columns)
    col_add.remove('ID')
    col += col_add
    print(df.shape)

    df_add = pd.read_csv('data/feat_ts_pred500.csv')
    df = pd.merge(df, df_add, on='ID', left_index=True, how='outer')
    col_add = list(df_add.columns)
    col_add.remove('ID')
    col += col_add
    print(df.shape)

    df_add = pd.read_csv('data/feat_pca.csv')
    df = pd.merge(df, df_add, on='ID', left_index=True, how='outer')
    col_add = list(df_add.columns)
    col_add.remove('ID')
    col += col_add
    print(df.shape)

    df_add = pd.read_csv('data/feat_pca_cores.csv')
    df = pd.merge(df, df_add, on='ID', left_index=True, how='outer')
    col_add = list(df_add.columns)
    col_add.remove('ID')
    col += col_add
    print(df.shape)

    df_add = pd.read_csv('data/feat_svd_cores.csv')
    df = pd.merge(df, df_add, on='ID', left_index=True, how='outer')
    col_add = list(df_add.columns)
    col_add.remove('ID')
    col += col_add
    print(df.shape)

    df_add = pd.read_csv('data/feat_simple_predict.csv')
    df = pd.merge(df, df_add, on='ID', left_index=True, how='outer')
    col_add = list(df_add.columns)
    col_add.remove('ID')
    col += col_add
    print(df.shape)

    df.fillna(0, inplace=True)

    #train = df[(df.target > 0) & (df.is_train == 0)]
    train = df[(df.target > 0)]
    test = df[(df.is_train == 0)]
    test_ref = df[(df.is_train == 0)]

    out = [feat['feature'] for feat in eli5_feats['feature_importances']['importances'] if feat['weight'] < 0.0007]

    col = [c for c in col if c not in out]

    train = train[col + ['ID', 'target']]

    test.target = 0.0
    test = test[col + ['ID', 'target']]

    test_ref = test_ref[['ID', 'target']]

    return col, train, test, test_ref



def process_xgb():
    col, train, test, test_ref = load_data()
    print(train.shape, test.shape, test_ref.shape)

    params = {
        'colsample_bytree': 0.055,
        'colsample_bylevel': 0.4,
        'gamma': 1.5,
        'learning_rate': 0.01,
        'max_depth': 5,
        'objective': 'reg:linear',
        'booster': 'gbtree',
        'min_child_weight': 10,
        'n_estimators': 1800,
        'reg_alpha': 0,
        'reg_lambda': 0,
        'eval_metric': 'rmse',
        'subsample': 0.7,
        'silent': True,
        'seed': 7,
    }
    folds = 20
    full_score = 0.0
    xg_test = xgb.DMatrix(test[col])
    use_regressor = True
    use_regressor = False
    for fold in range(folds):
        x1, x2, y1, y2 = model_selection.train_test_split(train[col], np.log1p(train.target.values), test_size=0.0010, random_state=fold)

        if use_regressor:
            p = params
            model = xgb.XGBRegressor(colsample_bytree=p['colsample_bytree'], colsample_bylevel=p['colsample_bylevel'], gamma=p['gamma'], learning_rate=p['learning_rate'], max_depth=p['max_depth'], objective=p['objective'], booster=p['booster'], min_child_weight=p['min_child_weight'], n_estimators=p['n_estimators'], reg_alpha=p['reg_alpha'], reg_lambda=p['reg_lambda'], eval_metric=p['eval_metric'] , subsample=p['subsample'], silent=1, n_jobs = -1, early_stopping_rounds = 100, random_state=7, nthread=-1)
            model.fit(x1, y1)
            score = np.sqrt(mean_squared_error(y2, model.predict(x2)))
            test['target'] += np.expm1(model.predict(test[col]))
        else:
            xg_valid = xgb.DMatrix(x2, label=y2)
            xg_train = xgb.DMatrix(x1, label=y1)
            model = xgb.train(params, xg_train, params['n_estimators'])
            score = np.sqrt(mean_squared_error(y2, model.predict(xg_valid)))
            test['target'] += np.expm1(model.predict(xg_test))

        print('Fold', fold, 'Score', score)
        full_score += score

    full_score /= folds
    print('Full score', full_score)

    test['target'] /= folds

    test.loc[test_ref.target > 0, 'target'] = test_ref[test_ref.target > 0].target.values

    test[['ID', 'target']].to_csv('subxgb.csv', index=False)

    explain=False
    #explain=True
    if explain and not use_regressor:
        print(eli5.format_as_text(eli5.explain_weights(model, top=200)))


def main():
    preprocess()
    process_xgb()

if __name__ == '__main__':
    main()

