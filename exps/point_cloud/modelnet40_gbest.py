import os
import random
import numpy as np
import argparse
import logging
import pickle
from pprint import pformat

from exps.data import get_modelnet40_data_fps
from settree.set_data import SetDataset, OPERATIONS, flatten_datasets
import exps.eval_utils as eval



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='test')
    parser.add_argument("--log", action='store_true')
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()
    log_dir = os.path.join(os.path.abspath('__file__' + '/../'), 'outputs', 'fps')
    eval.create_logger(log_dir=log_dir,
                       log_name=args.exp_name,
                       dump=args.log)
    logging.info('Args:\n' + pformat(vars(args)))

    np.random.seed(args.seed)
    random.seed(args.seed)

    # x_train, y_train, x_test, y_test = get_modelnet40_data(down_sample=10,
    #                                                        do_standardize=True,
    #                                                        flip=False,
    #                                                        seed=args.seed)
    x_train, y_train, x_test, y_test = get_modelnet40_data_fps()

    ds_train = SetDataset(records=x_train, is_init=True)
    ds_test = SetDataset(records=x_test, is_init=True)

    logging.info(args)
    shared_gbdt_params = {'n_estimators': 150,
                          'learning_rate': 0.1,
                          'max_depth': 6,
                          'max_features': None,
                          'subsample': 1,
                          'random_state': args.seed}

    set_params = {'n_estimators': shared_gbdt_params['n_estimators'],
                  'operations': OPERATIONS,
                  'splitter': 'sklearn',
                  'use_attention_set': True,
                  'use_attention_set_comp': False,
                  'attention_set_limit': 5,
                  'max_depth': shared_gbdt_params['max_depth'],
                  'max_features': shared_gbdt_params['max_features'],
                  'subsample': shared_gbdt_params['subsample'],
                  'random_state': shared_gbdt_params['random_state'],
                  'save_path': os.path.join(log_dir, '{}_checkpoint.pkl'.format(args.exp_name)),
                  'validation_fraction': 0.1,
                  'tol': 1e-3,
                  'n_iter_no_change': 5,
                  'verbose': 3}

    sklearn_params = {'n_estimators': shared_gbdt_params['n_estimators'],
                      'criterion': 'mse',
                      'learning_rate': shared_gbdt_params['learning_rate'],
                      'max_depth': shared_gbdt_params['max_depth'],
                      'max_features': shared_gbdt_params['max_features'],
                      'subsample': shared_gbdt_params['subsample'],
                      'validation_fraction': 0.1,
                      'tol': 1e-3,
                      'n_iter_no_change': 5,
                      'verbose': 3,
                      'random_state': shared_gbdt_params['random_state']}

    xgboost_params = {#'objective': 'binary:logistic', # 'multi:softmax', binary:logistic
                     'max_depth': shared_gbdt_params['max_depth'],
                     'n_jobs': 10,
                     'learning_rate': shared_gbdt_params['learning_rate'],
                     'n_estimators': shared_gbdt_params['n_estimators'],
                     'colsample_bytree': shared_gbdt_params['max_features'],
                     'subsample': shared_gbdt_params['subsample'],
                     'reg_lambda': 0,
                     'reg_alpha': 0,
                     'verbosity': 0,
                     'random_state': shared_gbdt_params['random_state'],
                     'seed': shared_gbdt_params['random_state']}

    x_train, x_test = flatten_datasets(ds_train, ds_test,
                                       operations_list=set_params['operations'],
                                       ds_val=None)

    xgboost_gbtd = eval.train_and_predict_xgboost(xgboost_params,
                                                  x_train, y_train,
                                                  x_test, y_test,
                                                  val_x=None, val_y=None,
                                                  early_stopping_rounds=None,
                                                  mode='multi_cls')

    set_gbtd = eval.train_and_predict_set_gbdt(set_params,
                                               ds_train, y_train,
                                               ds_test, y_test,
                                               eval_train=False,
                                               mode='multi_cls')

    if args.save:
        pkl_filename = os.path.join(log_dir, '{}_model.pkl'.format(args.exp_name))
        with open(pkl_filename, 'wb') as file:
            pickle.dump(set_gbtd, file)
