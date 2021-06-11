import os
import functools
import datetime
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split, Dataset
import torch
import torch.functional as F
import numpy as np


def get_datetime():
    return datetime.datetime.now()


def permute_set(records):
    shuffled = []
    for r in records:
        inds = np.arange(len(r))
        np.random.shuffle(inds)
        shuffled.append(r.take(inds, axis=0))
    return shuffled


def collate_func(batch, y_tensor=torch.LongTensor):
    data = []
    lens = []
    ys = []

    for item in batch:
        ys.append(item[1])
        lens.append(item[0].shape[0])
        data.append(item[0])

    data = torch.cat(data, axis=0)
    lens = tuple(lens)
    ys = y_tensor(ys)
    return data, lens, ys


def collate_func_rnn(batch, y_tensor=torch.LongTensor):
    data = []
    ys = []

    for item in batch:
        ys.append(item[1])
        data.append(item[0])

    data = pad_sequence(data, padding_value=0.0, batch_first=True)
    ys = y_tensor(ys)
    return data, ys


def collate_func_rnn_regression(batch, y_tensor=torch.FloatTensor):
    data = []
    ys = []

    for item in batch:
        ys.append(item[1])
        data.append(item[0])

    data = pad_sequence(data, padding_value=0.0, batch_first=True)
    ys = y_tensor(ys).reshape(-1, 1)
    return data, ys


def collate_func_equivariant(batch, y_tensor=torch.LongTensor):
    data = []
    lens = []
    ys = []

    for item in batch:
        ys.append(item[1])
        lens.append(item[0].shape[0])
        data.append(item[0])

    data = torch.cat(data, axis=0)
    ys = torch.cat(ys, axis=0)
    lens = tuple(lens)
    return data, lens, ys

def get_stats(list_of_records, out_tensor=True):
    X = np.concatenate(list_of_records)
    X_unique = np.unique(X, axis=0, return_index=False)
    m = X_unique.mean(0)
    s = X_unique.std(0)
    if out_tensor:
        m = torch.FloatTensor(m)
        s = torch.FloatTensor(s)
    return m, s


def feature_standart(x):
    return F.normalize(x, p=2, dim=1)


def normalize(m, s, eps=1e-8):
    def norm(m, s, x):
        return (x - m) / s
    s = s + eps
    return functools.partial(norm, m, s)


class DummyFlatDataset(Dataset):
    def __init__(self, x, y, transform_x=lambda x: x):
        self.x = torch.stack([torch.FloatTensor(i.flatten()) for i in x])
        self.x = transform_x(self.x)
        self.y = torch.FloatTensor(y)

    def __getitem__(self, i):
        return self.x[i, :], self.y[i]

    def __len__(self):
        return len(self.y)


class DummyDataset(Dataset):
    def __init__(self, x, y, transform_x=lambda x: x):
        self.x = [transform_x(torch.FloatTensor(i)) for i in x]
        self.y = torch.LongTensor(y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.y)


class DummyRegressionDataset(Dataset):
    def __init__(self, x, y, transform_x=lambda x: x):
        self.x = [transform_x(torch.FloatTensor(i)) for i in x]
        self.y = torch.FloatTensor(y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.y)


class DummyDummyDataset(Dataset):
    def __init__(self, x, y, transform_x=lambda x: x):
        self.x = [transform_x(torch.FloatTensor(i)) for i in x]
        self.y = torch.LongTensor(y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.y)

class EquivariantDummyDataset(Dataset):
    def __init__(self, x, y, transform_x=lambda x: x):
        self.x = [transform_x(torch.FloatTensor(i)) for i in x]
        self.y = [torch.FloatTensor(i) for i in y]

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.y)


def to_dataloader(x, y, dataset_type, batch_size, num_workers, shuffle=True, transform_x=lambda x: x, collate_fn=None):
    dataset = dataset_type(x, y, transform_x)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)


