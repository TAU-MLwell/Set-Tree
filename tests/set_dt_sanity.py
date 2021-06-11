import os
import numpy as np
import random
import unittest
from timeit import default_timer as timer
from datetime import timedelta
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_openml

from settree.set_data import SetDataset, set_object_to_matrix
from settree.set_tree import SetTree
from settree.set_rf import SetRandomForestClassifier
from exps.eval_utils import split_to_random_sets


def get_first_quarter_data(num_samples, min_items_set=2, max_items_set=10, dim=2):

    def inject_samples_in_first_quarter(set_of_samples, min=1, max=1, dim=2):
        num = random.choice(range(min, max + 1))
        pos_points = np.random.uniform(low=0, high=1, size=(num, dim))
        set_of_samples[:num, :] = pos_points
        return set_of_samples

    def sample_point_not_from_first_quarter(dim=2):

        # sample a quarter (not the first)
        while True:
            r = np.random.normal(0, 1, dim) > 0
            if sum(r) < dim:
                break

        # sample a point from the quarter
        p = []
        for i in r:
            # pos
            if i:
                p.append(np.random.uniform(low=0, high=1))
            # neg
            else:
                p.append(np.random.uniform(low=-1, high=0))
        return tuple(p)

    def sample_set(num, dim):
        return np.stack([sample_point_not_from_first_quarter(dim) for _ in range(num)])

    s_1 = [sample_set(random.choice(range(min_items_set, max_items_set)), dim) for _ in range(num_samples // 2)]
    s_2 = [sample_set(random.choice(range(min_items_set, max_items_set)), dim) for _ in range(num_samples // 2)]
    s_2 = [inject_samples_in_first_quarter(i, min=1, max=1, dim=dim) for i in s_2]

    data = s_1 + s_2
    y = np.concatenate([np.zeros(len(s_1)), np.ones(len(s_2))]).astype(int)

    indx = np.arange(len(y))
    random.shuffle(indx)
    return [data[i] for i in indx], y[indx]


def get_data_rect_vs_diag(num_samples, min_items_set, max_items_set, dim=2):
    def sample_set_rect(set_size, arange=(0,1), dim=2):
        return np.random.uniform(low=arange[0], high=arange[1], size=(set_size, dim))

    def sample_set_diag(set_size, arange=(0,1), dim=2):
        p = np.random.uniform(low=arange[0], high=arange[1], size=set_size)
        return np.repeat(p.reshape(-1, 1), dim, axis=1)

    s_1 = [sample_set_rect(random.choice(range(min_items_set, max_items_set)), (0, 1), dim)
           for _ in range(num_samples // 2)]
    s_2 = [sample_set_diag(random.choice(range(min_items_set, max_items_set)), (0, 1), dim)
           for _ in range(num_samples // 2)]

    data = s_1 + s_2
    y = np.concatenate([np.zeros(len(s_1)), np.ones(len(s_2))]).astype(int)

    indx = np.arange(len(y))
    random.shuffle(indx)
    return np.array(data)[indx].tolist(), y[indx]


class TestToyProblems(unittest.TestCase):
    test_counter = 1

    def __init__(self, splitter='set', use_attention_set=True, attention_set_limit=1, use_attention_set_comp=True):
        self.tree_args = {'splitter': splitter,
                          'use_attention_set': use_attention_set,
                          'use_attention_set_comp': use_attention_set_comp,
                          'attention_set_limit': attention_set_limit}
        print('Test args: {}'.format(self.tree_args))

    def init(self, name):
        np.random.seed(42)
        random.seed(42)

        print('####################({})####################'.format(self.test_counter))
        print('Start test: {}'.format(name))
        self.test_counter += 1

    def start_timer(self):
        self.start = timer()

    def end_timer(self):
        end = timer()
        print('Time: {}'.format(timedelta(seconds=end - self.start)))

    def end(self):
        print('############################################\n')

    def first_quarter(self):
        self.init('first_quarter')
        set_size = 10
        train_data, train_y = get_first_quarter_data(num_samples=1000,
                                                     min_items_set=set_size,
                                                     max_items_set=set_size + 1,
                                                     dim=2)
        test_data, test_y = get_first_quarter_data(num_samples=1000,
                                                   min_items_set=set_size,
                                                   max_items_set=set_size + 1,
                                                   dim=2)
        ds_train = SetDataset(records=train_data, is_init=True)
        ds_test = SetDataset(records=test_data, is_init=True)

        dt = SetTree(**self.tree_args)
        self.start_timer()
        dt.fit(ds_train, train_y)
        self.end_timer()

        train_acc = (dt.predict(ds_train) == train_y).mean()
        test_acc = (dt.predict(ds_test) == test_y).mean()
        print('Results : set_size={} | train acc {:.4f} | test acc : {:.4f}'.format(set_size,train_acc, test_acc))
        print(dt)

        self.end()
        self.assertGreaterEqual(test_acc, 0.95)

    def first_quarter_high_dim(self):
        self.init('first_quarter_high_dim')
        set_size = 10
        train_data, train_y = get_first_quarter_data(num_samples=5000,
                                                     min_items_set=set_size,
                                                     max_items_set=set_size + 1,
                                                     dim=4)
        test_data, test_y = get_first_quarter_data(num_samples=1000,
                                                   min_items_set=set_size,
                                                   max_items_set=set_size + 1,
                                                   dim=4)
        ds_train = SetDataset(records=train_data, is_init=True)
        ds_test = SetDataset(records=test_data, is_init=True)
        dt = SetTree(**self.tree_args)

        self.start_timer()
        dt.fit(ds_train, train_y)
        self.end_timer()

        train_acc = (dt.predict(ds_train) == train_y).mean()
        test_acc = (dt.predict(ds_test) == test_y).mean()
        print('Results : set_size={} | train acc {:.4f} | test acc : {:.4f}'.format(set_size, train_acc, test_acc))
        print(dt)

        self.end()
        self.assertGreaterEqual(test_acc, 0.9)

    def first_quarter_high_dim_varying_lengths(self):
        self.init('first_quarter_high_dim_varying_lengths')
        set_size = 10
        train_data, train_y = get_first_quarter_data(num_samples=5000,
                                                     min_items_set=5,
                                                     max_items_set=15,
                                                     dim=4)
        test_data, test_y = get_first_quarter_data(num_samples=1000,
                                                   min_items_set=5,
                                                   max_items_set=15,
                                                   dim=4)
        ds_train = SetDataset(records=train_data, is_init=True)
        ds_test = SetDataset(records=test_data, is_init=True)
        dt = SetTree(**self.tree_args)

        self.start_timer()
        dt.fit(ds_train, train_y)
        self.end_timer()

        train_acc = (dt.predict(ds_train) == train_y).mean()
        test_acc = (dt.predict(ds_test) == test_y).mean()
        print('Results : set_size={} | train acc {:.4f} | test acc : {:.4f}'.format(set_size, train_acc, test_acc))
        print(dt)

        self.end()
        self.assertGreaterEqual(test_acc, 0.95)

    def first_quarter_vs_sklearn(self):
        self.init('first_quarter_vs_sklearn')
        set_size = 10
        train_data, train_y = get_first_quarter_data(num_samples=1000,
                                                     min_items_set=set_size,
                                                     max_items_set=set_size + 1,
                                                     dim=2)
        test_data, test_y = get_first_quarter_data(num_samples=1000,
                                                   min_items_set=set_size,
                                                   max_items_set=set_size + 1,
                                                   dim=2)
        ds_train = SetDataset(records=train_data, is_init=True)
        ds_test = SetDataset(records=test_data, is_init=True)
        dt = SetTree(**self.tree_args)

        self.start_timer()
        dt.fit(ds_train, train_y)
        self.end_timer()

        train_acc = (dt.predict(ds_train) == train_y).mean()
        test_acc = (dt.predict(ds_test) == test_y).mean()
        print('Results : set_size={} | train acc {:.4f} | test acc : {:.4f}'.format(set_size, train_acc, test_acc))
        print(dt)

        sklearn_dt = DecisionTreeClassifier(criterion="entropy")
        sk_train_x = set_object_to_matrix(ds_train, dt.operations)
        sk_test_x = set_object_to_matrix(ds_test, dt.operations)

        sklearn_dt = sklearn_dt.fit(sk_train_x, train_y)
        sklearn_train_acc = (sklearn_dt.predict(sk_train_x) == train_y).mean()
        sklearn_test_acc = (sklearn_dt.predict(sk_test_x) == test_y).mean()
        print('Results sklearn: set_size={} | train acc {:.4f} | test acc : {:.4f}'.format(set_size, sklearn_train_acc, sklearn_test_acc))

        print('Tree structure (depth, n_nodes): sklearn: ({}, {}) setDT: ({}, {})'.format(sklearn_dt.get_depth(),
                                                                                          sklearn_dt.tree_.node_count,
                                                                                          dt.depth, dt.n_nodes))
        self.end()
        self.assertGreaterEqual(test_acc, sklearn_test_acc)

    def rect_vs_diagonal(self):
        self.init('rect_vs_diagonal')
        set_size=10
        train_data, train_y = get_data_rect_vs_diag(num_samples=1000,
                                                    min_items_set=set_size,
                                                    max_items_set=set_size + 1,
                                                    dim=2)
        test_data, test_y = get_data_rect_vs_diag(num_samples=1000,
                                                  min_items_set=set_size,
                                                  max_items_set=set_size + 1,
                                                  dim=2)
        ds_train = SetDataset(records=train_data, is_init=True)
        ds_test = SetDataset(records=test_data, is_init=True)
        dt = SetTree(**self.tree_args)

        self.start_timer()
        dt.fit(ds_train, train_y)
        self.end_timer()

        train_acc = (dt.predict(ds_train) == train_y).mean()
        test_acc = (dt.predict(ds_test) == test_y).mean()
        print('Results : set_size={} | train acc {:.4f} | test acc : {:.4f}'.format(set_size, train_acc, test_acc))
        print(dt)

        self.end()
        self.assertGreaterEqual(test_acc, 0.95)

    def rect_vs_diagonal_high_dim(self):
        self.init('rect_vs_diagonal_high_dim')
        set_size = 10
        train_data, train_y = get_data_rect_vs_diag(num_samples=5000,
                                                    min_items_set=set_size,
                                                    max_items_set=set_size + 1,
                                                    dim=8)
        test_data, test_y = get_data_rect_vs_diag(num_samples=1000,
                                                  min_items_set=set_size,
                                                  max_items_set=set_size + 1,
                                                  dim=8)
        ds_train = SetDataset(records=train_data, is_init=True)
        ds_test = SetDataset(records=test_data, is_init=True)
        dt = SetTree(**self.tree_args)

        self.start_timer()
        dt.fit(ds_train, train_y)
        self.end_timer()

        train_acc = (dt.predict(ds_train) == train_y).mean()
        test_acc = (dt.predict(ds_test) == test_y).mean()
        print('Results : set_size={} | train acc {:.4f} | test acc : {:.4f}'.format(set_size, train_acc, test_acc))
        print(dt)

        self.end()
        self.assertGreaterEqual(test_acc, 0.95)

    def rect_vs_diagonal_vs_sklearn(self):
        self.init('rect_vs_diagonal_vs_sklearn')
        set_size = 10
        train_data, train_y = get_data_rect_vs_diag(num_samples=1000,
                                                    min_items_set=set_size,
                                                    max_items_set=set_size + 1,
                                                    dim=2)
        test_data, test_y = get_data_rect_vs_diag(num_samples=1000,
                                                  min_items_set=set_size,
                                                  max_items_set=set_size + 1,
                                                  dim=2)
        ds_train = SetDataset(records=train_data, is_init=True)
        ds_test = SetDataset(records=test_data, is_init=True)
        dt = SetTree(**self.tree_args)

        self.start_timer()
        dt.fit(ds_train, train_y)
        self.end_timer()

        train_acc = (dt.predict(ds_train) == train_y).mean()
        test_acc = (dt.predict(ds_test) == test_y).mean()
        print('Results : set_size={} | train acc {:.4f} | test acc : {:.4f}'.format(set_size, train_acc, test_acc))
        print(dt)

        sklearn_dt = DecisionTreeClassifier(criterion="entropy")
        sk_train_x = set_object_to_matrix(ds_train, dt.operations)
        sk_test_x = set_object_to_matrix(ds_test, dt.operations)

        sklearn_dt = sklearn_dt.fit(sk_train_x, train_y)
        sklearn_train_acc = (sklearn_dt.predict(sk_train_x) == train_y).mean()
        sklearn_test_acc = (sklearn_dt.predict(sk_test_x) == test_y).mean()
        print('Results sklearn: set_size={} | train acc {:.4f} | test acc : {:.4f}'.format(set_size,
                                                                                           sklearn_train_acc,
                                                                                           sklearn_test_acc))

        print('Tree structure (depth, n_nodes): sklearn: ({}, {}) setDT: ({}, {})'.format(sklearn_dt.get_depth(),
                                                                                          sklearn_dt.tree_.node_count,
                                                                                          dt.depth, dt.n_nodes))
        self.end()
        self.assertGreaterEqual(test_acc, sklearn_test_acc)

    def classify_mnist(self):

        self.init('classify_mnist')
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True,
                            data_home=os.path.join(os.path.abspath('__file__' + '/../../'), 'data'))
        y = y.astype(int)
        X_0 = X[y == 0, :]
        X_1 = X[y == 9, :]
        X_2 = X[y == 8, :]
        X_3 = X[y == 6, :]

        X_0 = split_to_random_sets(X_0, min_size=2, max_size=30)
        X_1 = split_to_random_sets(X_1, min_size=2, max_size=30)
        X_2 = split_to_random_sets(X_2, min_size=2, max_size=30)
        X_3 = split_to_random_sets(X_3, min_size=2, max_size=30)
        split = int(((len(X_0) + len(X_1) + len(X_2) + len(X_3)) / 4) * 0.2)

        data = X_0[:split] + X_1[:split] + X_2[:split] + X_3[:split]
        train_y = np.array([0] * len(X_0[:split]) + [1] * len(X_1[:split]) + [2] * len(X_2[:split]) + [3] * len(X_3[:split]))
        ds_train = SetDataset(records=data, is_init=True)

        data = X_0[split:] + X_1[split:] + X_2[split:] + X_3[split:]
        test_y = np.array([0] * len(X_0[split:]) + [1] * len(X_1[split:]) + [2] * len(X_2[split:]) + [3] * len(X_3[split:]))
        ds_test = SetDataset(records=data)
        dt = SetTree(**self.tree_args)

        self.start_timer()
        dt.fit(ds_train, train_y)
        self.end_timer()

        train_acc = (dt.predict(ds_train) == train_y).mean()
        test_acc = (dt.predict(ds_test) == test_y).mean()
        print('Results : train acc {:.4f} | test acc : {:.4f}'.format(train_acc, test_acc))
        print(dt)

        self.end()
        self.assertGreaterEqual(test_acc, 0.93)

    def classify_mnist_rf(self):

        self.init('classify_mnist_rf')
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True,
                            data_home=os.path.join(os.path.abspath('__file__' + '/../../'), 'data'))
        y = y.astype(int)
        X_0 = X[y == 0, :]
        X_1 = X[y == 9, :]
        X_2 = X[y == 8, :]
        X_3 = X[y == 6, :]

        X_0 = split_to_random_sets(X_0, min_size=2, max_size=30)
        X_1 = split_to_random_sets(X_1, min_size=2, max_size=30)
        X_2 = split_to_random_sets(X_2, min_size=2, max_size=30)
        X_3 = split_to_random_sets(X_3, min_size=2, max_size=30)
        split = int(((len(X_0) + len(X_1) + len(X_2) + len(X_3)) / 4) * 0.2)

        data = X_0[:split] + X_1[:split] + X_2[:split] + X_3[:split]
        train_y = [0] * len(X_0[:split]) + [1] * len(X_1[:split]) + [2] * len(X_2[:split]) + [3] * len(X_3[:split])
        ds_train = SetDataset(records=data, is_init=True)

        data = X_0[split:] + X_1[split:] + X_2[split:] + X_3[split:]
        test_y = [0] * len(X_0[split:]) + [1] * len(X_1[split:]) + [2] * len(X_2[split:]) + [3] * len(X_3[split:])
        ds_test = SetDataset(records=data, is_init=True)

        dt = SetRandomForestClassifier(n_estimators=4,
                                       criterion="entropy",
                                       max_samples=0.5,
                                       max_depth=6,
                                       max_features="auto",
                                       splitter=self.tree_args['splitter'],
                                       use_active_set=self.tree_args['use_active_set'],
                                       active_set_limit=self.tree_args['active_set_limit'],
                                       bootstrap=True,
                                       n_jobs=4,
                                       random_state=None,
                                       verbose=3)

        self.start_timer()
        dt.fit(ds_train, train_y)
        self.end_timer()

        train_acc = (dt.predict(ds_train) == train_y).mean()
        test_acc = (dt.predict(ds_test) == test_y).mean()
        print('Results : train acc {:.4f} | test acc : {:.4f}'.format(train_acc, test_acc))
        print(dt)

        self.end()
        self.assertGreaterEqual(test_acc, 0.94)


if __name__ == '__main__':
    np.random.seed(42)

    toy_tests = TestToyProblems(splitter='sklearn',
                                use_attention_set=True,
                                use_attention_set_comp=True,
                                attention_set_limit=3)

    toy_tests.first_quarter()
    toy_tests.first_quarter_high_dim()
    toy_tests.first_quarter_high_dim_varying_lengths()
    toy_tests.first_quarter_vs_sklearn()
    toy_tests.rect_vs_diagonal()
    toy_tests.rect_vs_diagonal_high_dim()
    toy_tests.rect_vs_diagonal_vs_sklearn()
    toy_tests.classify_mnist()

    print('######## End tests ########')