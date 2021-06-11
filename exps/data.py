import os
import numpy as np
import random
import logging
import pickle
import h5py
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split

import awkward
from energyflow.datasets import qg_jets
from energyflow.utils import data_split, to_categorical

from settree.set_data import SetDataset
import exps.eval_utils as eval

ROOT_DATA = '/home/royhir/projects/data/'


########################################################################################################################
# EXP 1: Jets
########################################################################################################################

def remap_pids(events, pid_i=3):
    pid2float_mapping = {22: 1., 211: 2., -211: 3., 321: 4., -321: 5., 130: 6., 2112: 7., -2112: 8.,
                         2212: 9., -2212: 10., 11: 11., -11: 12., 13: 13., -13: 14.}

    events_shape = events.shape
    pids = events[:,:,pid_i].astype(int).reshape((events_shape[0]*events_shape[1]))
    events[:,:,pid_i] = np.asarray([pid2float_mapping.get(pid, 0) for pid in pids]).reshape(events_shape[:2])


def to_setdataset(X, use_pids=False):
    """ Arrange the dense zero-padded X into sparse list of np arrays.
        If use ID, add small noise eps to avoid numerical errors.
    """
    eps = 1e-7
    sparse_data_list = []
    for record in X:
        stop = np.argwhere(record[:, 0] != 0)[-1].item() + 1
        sparse_record = record[:stop, :]
        if use_pids:
            sparse_record[np.where(record[:stop, -1] == 0)] = eps
        sparse_data_list.append(sparse_record)
    return SetDataset(records=sparse_data_list, is_init=True)


def get_qg_datset(train=10000, val=5000, test=5000, use_pids=True, cache_dir=os.path.join(ROOT_DATA, 'physics', 'energyflow')):

    def unique_rows(X, y):
        X_unique, i = np.unique(X, axis=0, return_index=True)
        y_unique = y[i]
        return X_unique, y_unique

    # load data
    X, y = qg_jets.load(train + val + test, cache_dir=cache_dir)

    # convert labels to categorical
    Y = to_categorical(y, num_classes=2)
    logging.info('Loaded quark and gluon jets')

    # preprocess by centering jets and normalizing pts
    for x in X:
        mask = x[:,0] > 0
        yphi_avg = np.average(x[mask, 1:3], weights=x[mask,0], axis=0)
        x[mask, 1:3] -= yphi_avg
        x[mask, 0] /= x[:, 0].sum()

    # handle particle id channel
    if use_pids:
        remap_pids(X, pid_i=3)
    else:
        X = X[:, :, :3]
    logging.info('Finished preprocessing')

    # do train/val/test split
    (X_train, X_val, X_test, Y_train, Y_val, Y_test) = data_split(X, Y, train=train, val=val, test=test)
    X_train, Y_train = unique_rows(X_train, Y_train)
    X_val, Y_val = unique_rows(X_val, Y_val)
    X_test, Y_test = unique_rows(X_test, Y_test)
    logging.info('Done train/val/test split')

    ds_train = to_setdataset(X_train, use_pids)
    y_train = Y_train.argmax(1)

    ds_val = to_setdataset(X_val, use_pids)
    y_val = Y_val.argmax(1)

    ds_test = to_setdataset(X_test, use_pids)
    y_test = Y_test.argmax(1)

    return ds_train, y_train, ds_val, y_val, ds_test, y_test


def stack_arrays(a, keys, axis=-1):
    flat_arr = np.stack([a[k].flatten() for k in keys], axis=axis)
    return awkward.JaggedArray.fromcounts(a[keys[0]].counts, flat_arr)


def pad_array(a, maxlen, value=0., dtype='float32'):
    x = (np.ones((len(a), maxlen)) * value).astype(dtype)
    for idx, s in enumerate(a):
        if not len(s):
            continue
        trunc = s[:maxlen].astype(dtype)
        x[idx, :len(trunc)] = trunc
    return x


class ParticleNetDataset(object):

    def __init__(self, filepath, feature_dict={}, label='label', pad_len=100, data_format='channel_first'):
        self.filepath = filepath
        self.feature_dict = feature_dict
        if len(feature_dict) == 0:
            feature_dict['points'] = ['part_etarel', 'part_phirel']
            feature_dict['features'] = ['part_pt_log', 'part_e_log', 'part_etarel', 'part_phirel']
            feature_dict['mask'] = ['part_pt_log']
        self.label = label
        self.pad_len = pad_len
        assert data_format in ('channel_first', 'channel_last')
        self.stack_axis = 1 if data_format == 'channel_first' else -1
        self._values = {}
        self._label = None
        self._load()

    def _load(self):
        logging.info('Start loading file %s' % self.filepath)
        counts = None
        with awkward.load(self.filepath) as a:
            self._label = a[self.label]
            for k in self.feature_dict:
                cols = self.feature_dict[k]
                if not isinstance(cols, (list, tuple)):
                    cols = [cols]
                arrs = []
                for col in cols:
                    if counts is None:
                        counts = a[col].counts
                    else:
                        assert np.array_equal(counts, a[col].counts)
                    arrs.append(pad_array(a[col], self.pad_len))
                self._values[k] = np.stack(arrs, axis=self.stack_axis)
        logging.info('Finished loading file %s' % self.filepath)

    def __len__(self):
        return len(self._label)

    def __getitem__(self, key):
        if key == self.label:
            return self._label
        else:
            return self._values[key]

    @property
    def X(self):
        return self._values

    @property
    def y(self):
        return self._label

    def shuffle(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        shuffle_indices = np.arange(self.__len__())
        np.random.shuffle(shuffle_indices)
        for k in self._values:
            self._values[k] = self._values[k][shuffle_indices]
        self._label = self._label[shuffle_indices]

########################################################################################################################
# EXP 2: Point Clouds
########################################################################################################################

def get_modelnet40_data_fps(fname=os.path.join(ROOT_DATA, 'ModelNet40_cloud.h5'),
                            fps_pickle_path=os.path.join(ROOT_DATA, '100p_fps_subsample.pkl')):

    with h5py.File(fname, 'r') as f:
        train_label = np.array(f['tr_labels']).astype(np.int64)
        test_label = np.array(f['test_labels']).astype(np.int64)

    with open(fps_pickle_path, 'rb') as f:
        fps_dict = pickle.load(f)

    train_data = fps_dict['train']
    test_data = fps_dict['test']
    n_train = len(train_data)
    n_test = len(test_data)

    logging.info('Loaded {} train and {} test samples'.format(n_train, n_test))
    logging.info('Preprocess FPS params: {}'.format(str(fps_dict['params'])))
    logging.info('A record shape is {}'.format(train_data[0].shape))

    return train_data, train_label, test_data, test_label


def get_modelnet40_data(fname=os.path.join(ROOT_DATA, 'ModelNet40_cloud.h5'),
                        down_sample=10, do_standardize=True, flip=False, seed=0):

    def standardize(x):
        clipper = np.mean(np.abs(x), (1, 2), keepdims=True)
        z = np.clip(x, -100 * clipper, 100 * clipper)
        mean = np.mean(z, (1, 2), keepdims=True)
        std = np.std(z, (1, 2), keepdims=True)
        return (z - mean) / std

    np.random.seed(seed)

    with h5py.File(fname, 'r') as f:
        train_data = np.array(f['tr_cloud']).astype(np.float32)
        train_label = np.array(f['tr_labels']).astype(np.int64)
        test_data = np.array(f['test_cloud']).astype(np.float32)
        test_label = np.array(f['test_labels']).astype(np.int64)

    num_classes = np.max(train_label) + 1

    n_train = len(train_data)
    n_test = len(test_data)
    logging.info('Loaded {} train and {} test samples'.format(n_train, n_test))

    prep1 = standardize if do_standardize else lambda x: x

    # select the subset of points to use throughout beforehand
    perm = np.random.permutation(train_data.shape[1])[::down_sample]

    train_data = prep1(train_data[:, perm])
    test_data = prep1(test_data[:, perm])

    if flip:
        train_data = [i.T for i in train_data]
        test_data = [i.T for i in test_data]

    else:
        train_data = [i for i in train_data]
        test_data = [i for i in test_data]

    logging.info('Finish pre-process {} train samples and {} test samples'.format(len(train_data), len(test_data)))
    logging.info('A record shape is {}'.format(train_data[0].shape))
    return train_data, train_label, test_data, test_label

########################################################################################################################
# EXP 3: Drug Errors Predictions
########################################################################################################################

def get_mimic_data_fold(type, num, root_data='/home/royhir/projects/SetTrees/eval/mimic/outputs/'):
    return eval.load_pickle(os.path.join(root_data, type, 'seed={}_{}_dataset.pkl'.format(num, type)))


########################################################################################################################
# EXP 4: Poker hands prediction
########################################################################################################################

def process_record_set_tree(v):
    v1 = np.array([v[1], v[3], v[5], v[7], v[9]]).reshape(-1, 1)
    v2 = np.zeros((5, 4))
    for i, j in enumerate([0, 2, 4, 6, 8]):
        v2[i, int(v[j]) - 1] = 1.0
    return np.concatenate([v1, v2], 1)


def pre_process_set_tree(df):
    df.columns = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'Label']
    y = df.pop('Label').to_numpy().astype(np.int64)
    vals = df.values.astype(np.float32)
    x = [process_record_set_tree(v) for v in vals]
    return x, y


def pre_process_deepsets(df):
    df.columns = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'Label']
    y = df.pop('Label').to_numpy().astype(np.int64)
    vals = df.values.astype(np.float32)
    x = [v.reshape(5, 2) for v in vals]
    return x, y


def pre_process_all_permutations(df):
    df.columns = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'Label']
    y = df.pop('Label').to_numpy().astype(np.int64)
    vals = df.values.astype(np.float32)

    pers_vals = []
    col_indxs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    for col_perm in itertools.permutations(col_indxs):
        pers_vals.append(vals.take(list(sum(col_perm, ())), axis=1))
    x = np.concatenate(pers_vals)
    y = np.tile(y, 120)
    return x, y


def pre_process_numpy(df):
    df.columns = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'Label']
    y = df.pop('Label').to_numpy().astype(np.int64)
    x = df.values.astype(np.float32)
    return x, y


def get_poker_dataset(pre_process_method, data_root='/home/royhir/projects/data/poker/', scenerio='bin', n_test=None, seed=0):
    train = pd.read_csv(os.path.join(data_root, 'poker-hand-training-true.data'), header=None)
    test = pd.read_csv(os.path.join(data_root,'poker-hand-testing.data'), header=None)

    if n_test:
        _, test = train_test_split(test, test_size=n_test / 1e6, random_state=seed)

    if pre_process_method == 'set_tree':
        x_train, y_train = pre_process_set_tree(train)
        x_test, y_test = pre_process_set_tree(test)

    elif pre_process_method == 'deepsets':
        x_train, y_train = pre_process_deepsets(train)
        x_test, y_test = pre_process_deepsets(test)

    elif pre_process_method == 'mlp':
        x_train, y_train = pre_process_numpy(train)
        x_test, y_test = pre_process_numpy(test)

    else:
        raise ValueError('Invaid pre-process method: {}'.format(pre_process_method))

    if scenerio == 'bin':
        logging.info('Poker - binary classification')
        y_train[y_train != 0] = 1
        y_test[y_test != 0] = 1

    elif scenerio == 'multi':
        logging.info('Poker - multi-class classification')

    else:
        raise ValueError('Invalid scenerio {}'.format(scenerio))

    logging.info('There are {} train and {} test'.format(len(x_train), len(x_test)))
    return x_train, y_train, x_test, y_test

########################################################################################################################
# EXP 5: Redshift Estimation
########################################################################################################################

def get_redmapper_dataset_ii(is_deepsets, data_root='/home/royhir/projects/data/redmapper/processed_ii/', test_size=0.1, seed=42):

    file_name = 'redmapped_processed_for_deepsets_ii.pickle' if is_deepsets else 'redmapped_processed_ii.pickle'
    pickle_file_path = os.path.join(data_root, file_name)

    with open(pickle_file_path, 'rb') as f:
        d = pickle.load(f)

    records = d['records']
    labels = d['labels']
    logging.info('Load labels with valid fields: {}'.format(d['fields']))

    logging.info('Loaded {} total records'.format(len(labels)))
    train_inds, test_inds = train_test_split(np.arange(len(labels)),
                                             test_size=test_size,
                                             random_state=np.random.RandomState(seed) if seed != None else None)

    train_data = [records[i] for i in train_inds]
    test_data = [records[i] for i in test_inds]
    train_y = [labels[i] for i in train_inds]
    test_y = [labels[i] for i in test_inds]
    logging.info('{} train and {} test'.format(len(train_y), len(test_y)))

    return train_data, train_y, test_data, test_y

