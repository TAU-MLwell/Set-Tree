import os
import numpy as np
import random
import unittest
from timeit import default_timer as timer
from datetime import timedelta
from sklearn.metrics import roc_auc_score

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble._gb import GradientBoostingClassifier, GradientBoostingRegressor
import xgboost as xgb

from settree.set_data import SetDataset, OPERATIONS, flatten_datasets
from settree.gbest import GradientBoostedSetTreeClassifier, GradientBoostedSetTreeRegressor
import exps.eval_utils as eval


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


class TestGBTDProblems(unittest.TestCase):
    test_counter = 1

    def __init__(self, splitter='set', use_attention_set=True, use_attention_set_comp=True, attention_set_limit=1):
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

    def train_and_predict_xgboost(self, params,
                                  train_x, train_y, test_x, test_y, val_x=None, val_y=None,
                                  early_stopping_rounds=None, mode='bin_cls'):

        print('############ XGBoost ############ ')

        if mode == 'bin_cls':
            gbdt = xgb.XGBClassifier(**params)
            eval_met = eval.acc
            eval_met_name = 'acc'

        elif mode == 'multi_cls':
            gbdt = xgb.XGBClassifier(**params)
            eval_met = eval.acc
            eval_met_name = 'acc'
        else:
            gbdt = xgb.XGBRegressor(**params)
            eval_met = eval.mse
            eval_met_name = 'mse'

        timer = eval.Timer()
        if np.any(val_x):
            gbdt.fit(X=train_x,
                     y=train_y,
                     eval_set=[(val_x, val_y)],
                     early_stopping_rounds=early_stopping_rounds)
        else:
            gbdt.fit(train_x, train_y)
        print('Train took: {}'.format(timer.end()))

        timer = eval.Timer()
        train_preds = gbdt.predict(train_x)
        print('Eval train took: {}'.format(timer.end()))
        test_preds = gbdt.predict(test_x)

        train_met = eval_met(train_y, train_preds)
        test_met = eval_met(test_y, test_preds)

        if mode == 'bin_cls':
            train_proba = gbdt.predict_proba(train_x)[:, 1]
            test_proba = gbdt.predict_proba(test_x)[:, 1]

            train_auc = roc_auc_score(train_y, train_proba)
            test_auc = roc_auc_score(test_y, test_proba)

            print(
                'Results : train {} {:.4f} auc: {:.4f} | test {} : {:.4f} auc: {:.4f}'.format(eval_met_name, train_met,
                                                                                              train_auc, eval_met_name,
                                                                                              test_met, test_auc))
        else:
            print('Results : train {} {:.4f} | test {} : {:.4f}'.format(eval_met_name, train_met,
                                                                               eval_met_name, test_met))
        return train_met, test_met

    def train_and_predict_sklearn_gbtd(self, params,
                                       train_x, train_y, test_x, test_y,
                                       mode='bin_cls'):
        print('############ Sklearn ############ ')

        if mode == 'bin_cls':
            gbdt = GradientBoostingClassifier(**params)
            eval_met = eval.acc
            eval_met_name = 'acc'

        elif mode == 'multi_cls':
            gbdt = GradientBoostingClassifier(**params)
            eval_met = eval.acc
            eval_met_name = 'acc'
        else:
            gbdt = GradientBoostingRegressor(**params)
            eval_met = eval.mse
            eval_met_name = 'mse'

        timer = eval.Timer()
        gbdt.fit(train_x, train_y)
        print('Train took: {}'.format(timer.end()))

        if mode == 'bin_cls':
            timer = eval.Timer()
            train_raw_predictions = gbdt.decision_function(train_x)
            print('Eval train took: {}'.format(timer.end()))
            test_raw_predictions = gbdt.decision_function(test_x)

            train_encoded_labels = gbdt.loss_._raw_prediction_to_decision(train_raw_predictions)
            train_preds = gbdt.classes_.take(train_encoded_labels, axis=0)
            test_encoded_labels = gbdt.loss_._raw_prediction_to_decision(test_raw_predictions)
            test_preds = gbdt.classes_.take(test_encoded_labels, axis=0)

            train_met = eval_met(train_y, train_preds)
            test_met = eval_met(test_y, test_preds)

            train_probs = gbdt.loss_._raw_prediction_to_proba(train_raw_predictions)
            test_probs = gbdt.loss_._raw_prediction_to_proba(test_raw_predictions)

            train_auc = roc_auc_score(train_y, train_probs[:, 1])
            test_auc = roc_auc_score(test_y, test_probs[:, 1])

            print(
                'Results : train {} {:.4f} auc: {:.4f} | test {} : {:.4f} auc: {:.4f}'.format(eval_met_name, train_met,
                                                                                              train_auc, eval_met_name,
                                                                                              test_met, test_auc))
        else:
            timer = eval.Timer()
            train_preds = gbdt.predict(train_x)
            print('Eval train took: {}'.format(timer.end()))
            test_preds = gbdt.predict(test_x)
            train_met = eval_met(train_y, train_preds)
            test_met = eval_met(test_y, test_preds)

            print('Results : train {} {:.4f} | test {} : {:.4f}'.format(eval_met_name, train_met,
                                                                               eval_met_name, test_met))
        return train_met, test_met

    def first_quarter_four_dim(self):
        self.init('first_quarter_four_dim')
        set_size = 10
        train_data, train_y = get_first_quarter_data(num_samples=2000,
                                                     min_items_set=set_size,
                                                     max_items_set=set_size + 1,
                                                     dim=4)
        test_data, test_y = get_first_quarter_data(num_samples=1000,
                                                   min_items_set=set_size,
                                                   max_items_set=set_size + 1,
                                                   dim=4)
        ds_train = SetDataset(records=train_data, is_init=True)
        ds_test = SetDataset(records=test_data, is_init=True)

        set_gbdt = GradientBoostedSetTreeClassifier(n_estimators=5,
                                                    operations=OPERATIONS,
                                                    splitter=self.tree_args['splitter'],
                                                    use_attention_set=self.tree_args['use_attention_set'],
                                                    use_attention_set_comp=self.tree_args['use_attention_set_comp'],
                                                    attention_set_limit=self.tree_args['attention_set_limit'],
                                                    max_depth=6,
                                                    max_features=4,
                                                    # n_iter_no_change=3,
                                                    # tol=1e-4,
                                                    subsample=0.5,
                                                    random_state=0,
                                                    verbose=3)

        self.start_timer()
        set_gbdt.fit(ds_train, train_y)
        self.end_timer()

        train_acc = (set_gbdt.predict(ds_train) == train_y).mean()
        set_test_acc = (set_gbdt.predict(ds_test) == test_y).mean()
        print('Results : set_size={} | train acc {:.4f} | test acc : {:.4f}'.format(set_size,train_acc, set_test_acc))

        self.end()
        self.assertGreaterEqual(set_test_acc, 0.05)

    def first_quarter_no_active_set_compare(self):
        self.init('first_quarter_no_active_set_compare')
        set_size = 10
        train_data, train_y = get_first_quarter_data(num_samples=2000,
                                                     min_items_set=set_size,
                                                     max_items_set=set_size + 1,
                                                     dim=4)
        test_data, test_y = get_first_quarter_data(num_samples=1000,
                                                   min_items_set=set_size,
                                                   max_items_set=set_size + 1,
                                                   dim=4)
        ds_train = SetDataset(records=train_data, is_init=True)
        ds_test = SetDataset(records=test_data, is_init=True)

        set_gbdt = GradientBoostedSetTreeClassifier(n_estimators=5,
                                                    operations=OPERATIONS,
                                                    splitter=self.tree_args['splitter'],
                                                    use_attention_set=False,
                                                    use_attention_set_comp=False,
                                                    attention_set_limit=1,
                                                    max_depth=6,
                                                    max_features=None,
                                                    subsample=1,
                                                    random_state=0,
                                                    verbose=3)

        self.start_timer()
        set_gbdt.fit(ds_train, train_y)
        self.end_timer()

        train_acc = (set_gbdt.predict(ds_train) == train_y).mean()
        set_test_acc = (set_gbdt.predict(ds_test) == test_y).mean()
        print('Results : set_size={} | train acc {:.4f} | test acc : {:.4f}'.format(set_size,train_acc, set_test_acc))

        train_x, test_x = flatten_datasets(ds_train, ds_test, operations_list=set_gbdt.operations)

        params = {'n_estimators': 5, 'learning_rate': 0.1, 'max_depth': 6, 'max_features': None,
                  'subsample': 1, 'criterion': 'mse', 'random_state': 42}
        sk_learn_train_acc, sk_learn_test_acc = self.train_and_predict_sklearn_gbtd(params, train_x, train_y,
                                                                                    test_x, test_y, mode='bin_cls')

        params = {'objective': 'binary:logistic', 'max_depth': 6, 'n_jobs': 0, 'eval_metric': ['error'],
                  'learning_rate': 0.1, 'n_estimators': 5, 'colsample_bytree': None, 'subsample': None,
                   'reg_lambda': 1, 'verbosity': 0, 'random_state': 0, 'seed': 0}
        self.train_and_predict_xgboost(params, train_x, train_y, test_x, test_y, val_x=None, val_y=None,
                                       early_stopping_rounds=None, mode='bin_cls')

        self.end()
        self.assertGreaterEqual(set_test_acc, sk_learn_test_acc)

    def influence_of_trees_depth(self):
        self.init('influence_of_trees_depth')
        set_size = 10
        train_data, train_y = get_first_quarter_data(num_samples=1000,
                                                     min_items_set=set_size,
                                                     max_items_set=set_size + 1,
                                                     dim=4)
        test_data, test_y = get_first_quarter_data(num_samples=200,
                                                   min_items_set=set_size,
                                                   max_items_set=set_size + 1,
                                                   dim=4)
        ds_train = SetDataset(records=train_data, is_init=True)
        ds_test = SetDataset(records=test_data, is_init=True)

        for d in [2, 4, 6, 8, 10]:
            set_gbdt = GradientBoostedSetTreeClassifier(n_estimators=8,
                                                        splitter=self.tree_args['splitter'],
                                                        use_attention_set=self.tree_args['use_attention_set'],
                                                        use_attention_set_comp=self.tree_args['use_attention_set_comp'],
                                                        attention_set_limit=self.tree_args['attention_set_limit'],
                                                        max_depth=d,
                                                        max_features=4,
                                                        n_iter_no_change=3,
                                                        tol=1e-4,
                                                        subsample=0.5,
                                                        random_state=0,
                                                        verbose=3)
            self.start_timer()
            set_gbdt.fit(ds_train, train_y)
            self.end_timer()

            train_acc = (set_gbdt.predict(ds_train) == train_y).mean()
            set_test_acc = (set_gbdt.predict(ds_test) == test_y).mean()
            print('Results depth:{}: set_size={} | train acc {:.4f} | test acc : {:.4f}'.format(d,
                                                                                                set_size, train_acc,
                                                                                                set_test_acc))

    def multiclass_mnist(self):

        self.init('multiclass_mnist')
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True,
                            data_home=os.path.join(os.path.abspath('__file__' + '/../../'), 'data'))
        y = y.astype(int)
        X_0 = X[y == 0, :]
        X_1 = X[y == 9, :]
        X_2 = X[y == 8, :]
        X_3 = X[y == 6, :]

        X_0 = eval.split_to_random_sets(X_0, min_size=2, max_size=30)
        X_1 = eval.split_to_random_sets(X_1, min_size=2, max_size=30)
        X_2 = eval.split_to_random_sets(X_2, min_size=2, max_size=30)
        X_3 = eval.split_to_random_sets(X_3, min_size=2, max_size=30)

        y = [0] * len(X_0) + [1] * len(X_1) + [2] * len(X_2) + [3] * len(X_3)
        X = X_0 + X_1 + X_2 + X_3
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
        ds_train = SetDataset(records=train_x, is_init=True)
        ds_test = SetDataset(records=test_x, is_init=True)

        set_gbdt = GradientBoostedSetTreeClassifier(n_estimators=5,
                                                    splitter=self.tree_args['splitter'],
                                                    use_attention_set=self.tree_args['use_attention_set'],
                                                    use_attention_set_comp=self.tree_args['use_attention_set_comp'],
                                                    attention_set_limit=self.tree_args['attention_set_limit'],
                                                    max_depth=2,
                                                    max_features=None,
                                                    n_iter_no_change=3,
                                                    tol=1e-4,
                                                    subsample=0.5,
                                                    random_state=0,
                                                    verbose=3)

        self.start_timer()
        set_gbdt.fit(ds_train, train_y)
        self.end_timer()

        train_acc = (set_gbdt.predict(ds_train) == train_y).mean()
        set_test_acc = (set_gbdt.predict(ds_test) == test_y).mean()
        print('Results : train acc {:.4f} | test acc : {:.4f}'.format(train_acc, set_test_acc))

        self.end()
        self.assertGreaterEqual(set_test_acc, 0.94)


if __name__ == '__main__':
    toy_tests = TestGBTDProblems(splitter='sklearn',
                                 use_attention_set=True,
                                 use_attention_set_comp=True,
                                 attention_set_limit=3)

    toy_tests.first_quarter_four_dim()
    toy_tests.first_quarter_no_active_set_compare()
    toy_tests.influence_of_trees_depth()
    toy_tests.multiclass_mnist()
    print('######## End tests ########')