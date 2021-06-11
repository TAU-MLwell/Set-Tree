import numpy as np
import random

from settree.set_data import SetDataset

########################################################################################################################
# EXP 1: First quarter
########################################################################################################################

def get_first_quarter_data(num_samples, min_items_set=2, max_items_set=10, dim=2):

    def inject_samples_in_first_quarter(set_of_samples, min=1, max=1, dim=2):
        num = random.choice(range(min, max + 1))
        pos_points = np.random.uniform(low=0, high=1, size=(num, dim))
        set_of_samples[:num, :] = pos_points
        return set_of_samples

    def sample_point_not_from_first_quarter(dim=2):
        # sample a quarter (not the first)
        while True:
            r = np.random.uniform(-1, 1, dim)
            if sum(r >= 0) < dim:
                break
        return tuple(r)

    def sample_set(num, dim):
        return np.stack([sample_point_not_from_first_quarter(dim) for _ in range(num)])

    s_1 = [sample_set(random.choice(range(min_items_set, max_items_set)), dim) for _ in range(num_samples // 2)]
    s_2 = [sample_set(random.choice(range(min_items_set, max_items_set)), dim) for _ in range(num_samples // 2)]
    s_2 = [inject_samples_in_first_quarter(i, min=1, max=1, dim=dim) for i in s_2]

    x = s_1 + s_2
    y = np.concatenate([np.zeros(len(s_1)), np.ones(len(s_2))]).astype(np.int64)
    return x, y

########################################################################################################################
# EXP 2: Stats
########################################################################################################################

def get_data_uniform_vs_normal(n, set_size):
    neg = []
    for _ in range(n//2):
        if np.random.rand() > 0.5:
            ids = np.array([0] * (set_size // 2) + [1] * (set_size // 2))
        else:
            ids = np.array([0] * (set_size // 2) + [1] * (set_size // 2))
        neg.append(np.stack([np.concatenate([np.random.normal(loc=0.0, scale=1.0, size=(set_size // 2,)),
                                   np.random.uniform(low=-1.0, high=1.0, size=(set_size // 2,))]), ids], axis=1))

    pos = []
    for _ in range(n//4):
        if np.random.rand() > 0.5:
            ids = np.array([0] * (set_size // 2) + [1] * (set_size // 2))
        else:
            ids = np.array([0] * (set_size // 2) + [1] * (set_size // 2))

        pos.append(np.stack([np.random.normal(loc=0.0, scale=1.0, size=(set_size,)), ids], axis=1))
        pos.append(np.stack([np.random.uniform(low=-1.0, high=1.0, size=(set_size,)), ids], axis=1))

    y = np.array([0] * (n // 2) + [1] * (n // 2))
    x = pos + neg
    return x, y


def get_data_laplace_vs_normal(n, set_size):
    neg = []
    for _ in range(n//2):
        if np.random.rand() > 0.5:
            ids = np.array([0] * (set_size // 2) + [1] * (set_size // 2))
        else:
            ids = np.array([0] * (set_size // 2) + [1] * (set_size // 2))
        neg.append(np.stack([np.concatenate([np.random.normal(loc=0.0, scale=1.0, size=(set_size // 2,)),
                                   np.random.laplace(loc=0.0, scale=1.0, size=(set_size // 2,))]), ids], axis=1))

    pos = []
    for _ in range(n//4):
        if np.random.rand() > 0.5:
            ids = np.array([0] * (set_size // 2) + [1] * (set_size // 2))
        else:
            ids = np.array([0] * (set_size // 2) + [1] * (set_size // 2))

        pos.append(np.stack([np.random.normal(loc=0.0, scale=1.0, size=(set_size,)), ids], axis=1))
        pos.append(np.stack([np.random.laplace(loc=0.0, scale=1.0, size=(set_size,)), ids], axis=1))

    y = np.array([0] * (n // 2) + [1] * (n // 2))
    x = pos + neg
    return x, y


def get_data_different_mu_normal(n, set_size):
    neg = []
    for _ in range(n//2):
        if np.random.rand() > 0.5:
            ids = np.array([0] * (set_size // 2) + [1] * (set_size // 2))
        else:
            ids = np.array([0] * (set_size // 2) + [1] * (set_size // 2))
        neg.append(np.stack([np.concatenate([np.random.normal(loc=np.random.randn(), scale=1.0, size=(set_size // 2,)),
                                   np.random.normal(loc=np.random.randn(), scale=1.0, size=(set_size // 2,))]), ids], axis=1))

    pos = []
    for _ in range(n//4):
        if np.random.rand() > 0.5:
            ids = np.array([0] * (set_size // 2) + [1] * (set_size // 2))
        else:
            ids = np.array([0] * (set_size // 2) + [1] * (set_size // 2))
        mu = np.random.randn()
        pos.append(np.stack([np.random.normal(loc=mu, scale=1.0, size=(set_size,)), ids], axis=1))
        pos.append(np.stack([np.random.normal(loc=mu, scale=1.0, size=(set_size,)), ids], axis=1))

    y = np.array([0] * (n // 2) + [1] * (n // 2))
    x = pos + neg
    return x, y


def get_data_different_sigma_normal(n, set_size):
    neg = []
    for _ in range(n//2):
        if np.random.rand() > 0.5:
            ids = np.array([0] * (set_size // 2) + [1] * (set_size // 2))
        else:
            ids = np.array([0] * (set_size // 2) + [1] * (set_size // 2))
        neg.append(np.stack([np.concatenate([np.random.normal(loc=0.0, scale=np.abs(np.random.randn()), size=(set_size // 2,)),
                                   np.random.normal(loc=0.0, scale=np.abs(np.random.randn()), size=(set_size // 2,))]), ids], axis=1))

    pos = []
    for _ in range(n//4):
        if np.random.rand() > 0.5:
            ids = np.array([0] * (set_size // 2) + [1] * (set_size // 2))
        else:
            ids = np.array([0] * (set_size // 2) + [1] * (set_size // 2))
        sig = np.abs(np.random.randn())
        pos.append(np.stack([np.random.normal(loc=0.0, scale=sig, size=(set_size,)), ids], axis=1))
        pos.append(np.stack([np.random.normal(loc=0.0, scale=sig, size=(set_size,)), ids], axis=1))

    y = np.array([0] * (n // 2) + [1] * (n // 2))
    x = pos + neg
    return x, y


def get_data_by_task(task_name, params):
    if task_name == 'different_uniform_normal':
        # 1) different distributions
        x_train, y_train = get_data_uniform_vs_normal(params['n_train'], params['set_size'])
        ds_train = SetDataset(records=x_train, is_init=True)
        x_test, y_test = get_data_uniform_vs_normal(params['n_test'], params['set_size'])
        ds_test = SetDataset(records=x_test, is_init=True)

    elif task_name == 'different_laplace_normal':
        # 1) different distributions
        x_train, y_train = get_data_laplace_vs_normal(params['n_train'], params['set_size'])
        ds_train = SetDataset(records=x_train, is_init=True)
        x_test, y_test = get_data_laplace_vs_normal(params['n_test'], params['set_size'])
        ds_test = SetDataset(records=x_test, is_init=True)

    elif task_name == 'different_mean':
        # 2) different mean
        x_train, y_train = get_data_different_mu_normal(params['n_train'], params['set_size'])
        ds_train = SetDataset(records=x_train, is_init=True)
        x_test, y_test = get_data_different_mu_normal(params['n_test'], params['set_size'])
        ds_test = SetDataset(records=x_test, is_init=True)

    elif task_name == 'different_std':
        # 3) different sigma
        x_train, y_train = get_data_different_sigma_normal(params['n_train'], params['set_size'])
        ds_train = SetDataset(records=x_train, is_init=True)
        x_test, y_test = get_data_different_sigma_normal(params['n_test'], params['set_size'])
        ds_test = SetDataset(records=x_test, is_init=True)

    else:
        raise ValueError
    return ds_train, y_train, ds_test, y_test

