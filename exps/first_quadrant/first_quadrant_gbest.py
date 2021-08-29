import os
import logging
import numpy as np

from settree.set_data import SetDataset, OPERATIONS, flatten_datasets
from exps.synthetic_data import get_first_quarter_data
import exps.eval_utils as eval


if __name__ == '__main__':

    params = {'exp_name': 'gbdt_100di',
              'seed': 0,
              'n_train': 100000,
              'n_test': 10000,
              'dim': 100,
              'train_set_size': 20,
              'test_sizes': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300],
              'n_exp': 5,

              'n_estimators': 50,
              'learning_rate': 0.1,
              'max_depth': 12,
              'max_features': None,
              'subsample': 0.5,
              'random_state': 0
              }

    log_dir = os.path.join(os.path.abspath('__file__' + '/../'), 'outputs', '100dim_multi')
    eval.create_logger(log_dir=log_dir,
                       log_name=params['exp_name'],
                       dump=True)

    exp2results = {}
    for seed in range(params['n_exp']):
        logging.info('Start exp {}'.format(seed))
        params['seed'] = seed
        np.random.seed(seed)

        logging.info('Start train: set_size={} dim={}'.format(params['n_train'], params['dim']))
        train_data, y_train = get_first_quarter_data(num_samples=params['n_train'],
                                                     min_items_set=params['train_set_size'],
                                                     max_items_set=params['train_set_size'] + 1,
                                                     dim=params['dim'])
        ds_train = SetDataset(records=train_data, is_init=True)
        test_data, y_test = get_first_quarter_data(num_samples=params['n_test'],
                                                   min_items_set=params['train_set_size'],
                                                   max_items_set=params['train_set_size'] + 1,
                                                   dim=params['dim'])
        ds_test = SetDataset(records=test_data, is_init=True)

        set_params = {'n_estimators': params['n_estimators'],
                      'operations': OPERATIONS,
                      'splitter': 'sklearn',
                      'use_attention_set': True,
                      'attention_set_limit': 10,
                      'max_depth': params['max_depth'],
                      'max_features': params['max_features'],
                      'subsample': params['subsample'],
                      'random_state': params['random_state'],
                      'validation_fraction': 0.1,
                      'tol': 1e-3,
                      'n_iter_no_change': 3,
                      'verbose': 3}

        xgboost_params = {'n_estimators': params['n_estimators'],
                          'criterion': 'mse',
                          'learning_rate': params['learning_rate'],
                          'max_depth': params['max_depth'],
                          'max_features': params['max_features'],
                          'subsample': params['subsample'],
                          'validation_fraction': 0.1,
                          'tol': 1e-3,
                          'n_iter_no_change': 5,
                          'verbose': 0,
                          'random_state': params['random_state']}

        x_train, x_test = flatten_datasets(ds_train, ds_test, operations_list=set_params['operations'], ds_val=None)
        reg_tree, train_acc, test_acc = eval.train_and_predict_xgboost(xgboost_params,
                                                                       x_train, y_train,
                                                                       x_test, y_test,
                                                                       mode='bin_cls',
                                                                       out_metrics=True, verbose=False)

        set_tree, set_train_acc, set_test_acc = eval.train_and_predict_set_gbdt(set_params,
                                                                                ds_train, y_train,
                                                                                ds_test, y_test,
                                                                                mode='bin_cls',
                                                                                out_metrics=True,
                                                                                verbose=False)

        reg_tree_acc = []
        set_tree_acc = []
        for test_set_size in params['test_sizes']:
            test_data, y_test = get_first_quarter_data(num_samples=params['n_test'],
                                                       min_items_set=test_set_size,
                                                       max_items_set=test_set_size + 1,
                                                       dim=params['dim'])
            ds_test = SetDataset(records=test_data, is_init=True)

            test_acc = eval.acc(reg_tree.predict(eval.set_object_to_matrix(ds_test, set_tree.operations)), y_test)
            set_test_acc = eval.acc(set_tree.predict(ds_test), y_test)
            logging.info('Test set size: {} | No AS: {:.5f} with AS: {:.5f}'.format(test_set_size, test_acc, set_test_acc))

            set_tree_acc.append(set_test_acc)
            reg_tree_acc.append(test_acc)
        exp2results[seed] = {'reg': reg_tree_acc, 'set': set_tree_acc}

    eval.save_json(exp2results, os.path.join(log_dir, '{}_results_dump.json'.format(params['exp_name'])))
