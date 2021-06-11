import os
import numpy as np
import argparse
import logging
import random
import pickle
from pprint import pformat

from exps.data import ParticleNetDataset
from settree.set_data import SetDataset, OPERATIONS, merge_init_datasets
import exps.eval_utils as eval
from exps.eval_utils import create_logger

data_root = '/home/royhir/projects/data/physics/top_quark/proc'


def pre_process(dataset, limit=None):
    x = dataset.X
    y = dataset.y

    if limit is None:
        limit = len(y)

    inds = random.sample(range(len(y)), limit)
    x_points = x['points'].take(inds, axis=0)
    x_features = x['features'].take(inds, axis=0)
    x_mask = x['mask'].take(inds, axis=0)
    y = y.take(inds, axis=0)
    y = y.argmax(1)

    records = []
    ys = []
    for p, f, m, y in zip(x_points, x_features, x_mask, y):
        try:
            m_row = np.where(p.any(axis=1))[0].max()
            records.append(np.concatenate((p[:m_row, :], f[:m_row, :], m[:m_row, :]),axis=1))
            ys.append(y)
        except:
            pass
    return records, np.array(ys)


def get_top_quark_datset(train=None, val=None, test=None):
    train_dataset = ParticleNetDataset(os.path.join(data_root, 'train_file_0.awkd'), data_format='channel_last')
    val_dataset = ParticleNetDataset(os.path.join(data_root, 'val_file_0.awkd'), data_format='channel_last')
    test_dataset = ParticleNetDataset(os.path.join(data_root, 'test_file_0.awkd'), data_format='channel_last')
    logging.info('Loaded raw data')

    train_records, train_y = pre_process(train_dataset, limit=train)
    val_records, val_y = pre_process(val_dataset, limit=val)
    test_records, test_y = pre_process(test_dataset, limit=test)
    logging.info('Finish pre-processing')
    logging.info('train: {} val: {} test: {}'.format(len(train_y), len(val_y), len(test_y)))
    return SetDataset(records=train_records, is_init=True), train_y, \
           SetDataset(records=val_records, is_init=True), val_y, \
           SetDataset(records=test_records, is_init=True), test_y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='test')
    parser.add_argument("--splits", type=int, nargs="+", default=[1200000, 400000, 400000])
    parser.add_argument("--attention_set_limit", type=int, default=6)
    parser.add_argument("--use_attention_set", action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument("--log", action='store_true')

    args = parser.parse_args()

    np.random.seed(42)
    random.seed(42)
    log_dir = os.path.join(os.path.abspath('__file__' + '/../'), 'outputs', 'top_quark')
    create_logger(log_dir=log_dir,
                  log_name=args.exp_name,
                  dump=args.log)

    logging.info(args)
    train, val, test = args.splits
    ds_train, y_train, ds_val, y_val, ds_test, y_test = get_top_quark_datset(train, val, test)

    shared_gbdt_params = {'n_estimators': 50,
                          'learning_rate': 0.1,
                          'max_depth': 8,
                          'max_features': None,
                          'subsample': 0.5,
                          'criterion': 'mse',
                          'early_stopping_rounds': 5,
                          'random_state': 42}

    logging.info('Shared params:\n' + pformat(shared_gbdt_params))

    set_params = {'n_estimators': shared_gbdt_params['n_estimators'],
                  'operations': OPERATIONS,
                  'splitter': 'sklearn',
                  'use_attention_set': True,
                  'use_attention_set_comp': False,
                  'attention_set_limit': args.attention_set_limit,
                  'max_depth': shared_gbdt_params['max_depth'],
                  'max_features': shared_gbdt_params['max_features'],
                  'subsample': shared_gbdt_params['subsample'],
                  'random_state': shared_gbdt_params['random_state'],
                  'save_path': None,
                  'validation_fraction': 0.25,
                  'tol': 1e-4,
                  'n_iter_no_change': shared_gbdt_params['early_stopping_rounds'],
                  'verbose': 3}

    sklearn_params = {'n_estimators': shared_gbdt_params['n_estimators'],
                      'criterion': 'mse',
                      'learning_rate': shared_gbdt_params['learning_rate'],
                      'max_depth': shared_gbdt_params['max_depth'],
                      'max_features': shared_gbdt_params['max_features'],
                      'subsample': shared_gbdt_params['subsample'],
                      'validation_fraction': 0.25,
                      'tol': 1e-4,
                      'n_iter_no_change': shared_gbdt_params['early_stopping_rounds'],
                      'random_state': shared_gbdt_params['random_state']}

    xgboost_params = {#'tree_method': 'gpu_hist',
                      #'gpu_id': 7,
                      #'objective': 'binary:logistic',
                      'max_depth': shared_gbdt_params['max_depth'],
                      'n_jobs': 10,
                      'eval_metric': ['error'],
                      'learning_rate': shared_gbdt_params['learning_rate'],
                      'n_estimators': shared_gbdt_params['n_estimators'],
                      'colsample_bytree': shared_gbdt_params['max_features'],
                      'subsample': shared_gbdt_params['subsample'],
                      'reg_lambda': 0,
                      'verbosity': 0,
                      'random_state': shared_gbdt_params['random_state'],
                      'seed': shared_gbdt_params['random_state']}

    x_train, x_test, x_val = eval.flatten_datasets(ds_train, ds_test,
                                                   operations_list=set_params['operations'],
                                                   ds_val=ds_val)

    xgboost_gbtd = eval.train_and_predict_xgboost(xgboost_params,
                                                  x_train, y_train,
                                                  x_test, y_test,
                                                  val_x=None, val_y=None,
                                                  early_stopping_rounds=None)

    ds_train_val = merge_init_datasets(ds_train, ds_val)
    set_gbtd = eval.train_and_predict_set_gbdt(set_params,
                                               ds_train_val, np.concatenate([y_train, y_val]),
                                               ds_test, y_test,
                                               resume=None)

    if args.save:
        pkl_filename = os.path.join(log_dir, '{}_model.pkl'.format(args.exp_name))
        with open(pkl_filename, 'wb') as file:
            pickle.dump(set_gbtd, file)
