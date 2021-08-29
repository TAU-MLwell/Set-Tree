import pickle
import numpy as np
import os
import random
import argparse
import logging
from pprint import pformat

from settree.set_data import SetDataset, OPERATIONS, flatten_datasets
import exps.eval_utils as eval
from exps.data import get_redmapper_dataset_ii


def process_data_to_xgboost(train_data, test_data):
    def data_process(x):
        p_x = []
        for r in x:
            p_x.append(r[np.where(r[:, -1] == 1)[0].item(), :-1])
        return np.stack(p_x)

    return data_process(train_data), data_process(test_data)


def eval_scatter(model, x, y):
    y_pred = np.array(model.predict(x))
    y = np.array(y)
    return np.average((np.abs(y - y_pred)) / (1 + y))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='test', help="name of experiment")
    parser.add_argument("--splitter", type=str, default='sklearn')
    parser.add_argument("--attention_set_limit", type=int, default=6)
    parser.add_argument("--use_attention_set", action='store_true')

    parser.add_argument("--log", action='store_true')
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()

    log_dir = os.path.join(os.path.abspath('__file__' + '/../'), 'outputs', 'redmapper_ii')
    eval.create_logger(log_dir=log_dir,
                       log_name=args.exp_name,
                       dump=args.log)
    logging.info('Args:\n' + pformat(vars(args)))

    for seed in range(5):
        np.random.seed(seed)
        random.seed(seed)
        logging.info('Seed {}'.format(seed))

        train_data, train_y, test_data, test_y = get_redmapper_dataset_ii(is_deepsets=False, seed=seed)
        ds_train = SetDataset(records=train_data, is_init=True)
        ds_test = SetDataset(records=test_data, is_init=True)

        shared_gbdt_params = {'n_estimators': 50,
                              'learning_rate': 0.1,
                              'max_depth': 8,
                              'max_features': None,
                              'subsample': 0.5,
                              'criterion': 'mse',
                              #'early_stopping_rounds': 5,
                              'random_state': 42}

        set_params = {'n_estimators': shared_gbdt_params['n_estimators'],
                      'operations': OPERATIONS,
                      'splitter': args.splitter,
                      'use_attention_set': args.use_attention_set,
                      'attention_set_limit': args.attention_set_limit,
                      'max_depth': shared_gbdt_params['max_depth'],
                      'max_features': shared_gbdt_params['max_features'],
                      'subsample': shared_gbdt_params['subsample'],
                      'random_state': shared_gbdt_params['random_state'],
                      # 'validation_fraction': 0.11,
                      # 'tol': 1e-4,
                      # 'n_iter_no_change': shared_gbdt_params['early_stopping_rounds'],
                      'verbose': 3}

        xgboost_params = {#'tree_method': 'gpu_hist',
                          #'gpu_id': 7,
                          'objective': 'binary:logistic',
                          'max_depth': shared_gbdt_params['max_depth'],
                          'n_jobs': 0,
                          'eval_metric': ['error'],
                          'learning_rate': shared_gbdt_params['learning_rate'],
                          'n_estimators': shared_gbdt_params['n_estimators'],
                          'colsample_bytree': None,
                          'subsample': shared_gbdt_params['subsample'],
                          'reg_lambda': 0,
                          'verbosity': 0,
                          'random_state': shared_gbdt_params['random_state'],
                          'seed': shared_gbdt_params['random_state']}
        train_x, test_x = process_data_to_xgboost(train_data, test_data)
        xgboost_gbtd = eval.train_and_predict_xgboost(xgboost_params,
                                                      train_x, train_y, test_x, test_y, val_x=None, val_y=None,
                                                      early_stopping_rounds=None, mode='reg')
        logging.info('Train metric: {:.5f} test metric: {:.5f}'.format(eval_scatter(xgboost_gbtd, train_x, train_y),
                                                                           eval_scatter(xgboost_gbtd, test_x, test_y)))

        set_gbtd = eval.train_and_predict_set_gbdt(set_params, ds_train, train_y, ds_test, test_y, mode='reg')
        logging.info('Train metric: {:.5f} test metric: {:.5f}'.format(eval_scatter(set_gbtd, ds_train, train_y),
                                                                       eval_scatter(set_gbtd, ds_test, test_y)))

    if args.save:
        pkl_filename = os.path.join(log_dir, '{}_model.pkl'.format(args.exp_name))
        with open(pkl_filename, 'wb') as file:
            pickle.dump(set_gbtd, file)
