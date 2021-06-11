import os
import random
import numpy as np
import argparse
import logging
import pickle
from pprint import pformat

from settree.set_data import OPERATIONS, merge_init_datasets, flatten_datasets
import exps.eval_utils as eval
from exps.data import get_qg_datset


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='test_new_gbtd', help="name of experiment")
    parser.add_argument("--splits", type=int, nargs="+", default=[1600000, 200000, 200000], help="data splits")
    parser.add_argument("--use_pids", action='store_true', help="if TRUE uses pids data")

    parser.add_argument("--splitter", type=str, default='sklearn')
    parser.add_argument("--attention_set_limit", type=int, default=3)
    parser.add_argument("--use_attention_set", action='store_true')

    parser.add_argument("--log", action='store_true')
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)

    args = parser.parse_args()

    log_dir = os.path.join(os.path.abspath('__file__' + '/../'), 'outputs', 'qg_jet')
    eval.create_logger(log_dir=log_dir,
                       log_name=args.exp_name,
                       dump=args.log)
    logging.info('Args:\n' + pformat(vars(args)))

    np.random.seed(args.seed)
    random.seed(args.seed)

    train, val, test = args.splits
    ds_train, y_train, ds_val, y_val, ds_test, y_test = get_qg_datset(train, val, test, args.use_pids)

    logging.info(args)
    if args.checkpoint:
        logging.info('Load checkpoint from {}'.format(args.checkpoint))
        gbdt = eval.load_checkpoint_gbdt(args.checkpoint)

        test_raw_predictions = gbdt.decision_function(ds_test)
        test_encoded_labels = gbdt.loss_._raw_prediction_to_decision(test_raw_predictions)
        test_preds = gbdt.classes_.take(test_encoded_labels, axis=0)
        test_probs = gbdt.loss_._raw_prediction_to_proba(test_raw_predictions)

        test_met = eval.acc(y_test, test_preds)
        test_auc = eval.roc_auc_score(y_test, test_probs[:, 1])
        logging.info('Results test : {:.4f} auc: {:.4f}'.format(test_met, test_auc))

    else:

        shared_gbdt_params = {'n_estimators': 150,
                              'learning_rate': 0.1,
                              'max_depth': 5,
                              'max_features': None,
                              'subsample': 0.5,
                              'criterion': 'mse',
                              'early_stopping_rounds': 5,
                              'random_state': args.seed}

        logging.info('Shared params:\n' + pformat(shared_gbdt_params))

        set_params = {'n_estimators': shared_gbdt_params['n_estimators'],
                      'operations': OPERATIONS,
                      'splitter': args.splitter,
                      'use_attention_set': args.use_attention_set,
                      'use_attention_set_comp': False,
                      'attention_set_limit': args.attention_set_limit,
                      'max_depth': shared_gbdt_params['max_depth'],
                      'max_features': shared_gbdt_params['max_features'],
                      'subsample': shared_gbdt_params['subsample'],
                      'random_state': shared_gbdt_params['random_state'],
                      'save_path': '/home/royhir/projects/SetTrees/eval/jets/outputs/qg_jet/exp_4_checkpoint.pkl', # TODO
                      'validation_fraction': 0.11,
                      'tol': 1e-4,
                      'n_iter_no_change': shared_gbdt_params['early_stopping_rounds'],
                      'verbose': 3}

        sklearn_params = {'n_estimators': shared_gbdt_params['n_estimators'],
                          'criterion': 'mse',
                          'learning_rate': shared_gbdt_params['learning_rate'],
                          'max_depth': shared_gbdt_params['max_depth'],
                          'max_features': shared_gbdt_params['max_features'],
                          'subsample': shared_gbdt_params['subsample'],
                          'validation_fraction': 0.11,
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

        x_train, x_test, x_val = flatten_datasets(ds_train, ds_test,
                                                  operations_list=set_params['operations'],
                                                  ds_val=ds_val)

        xgboost_gbtd = eval.train_and_predict_xgboost(xgboost_params,
                                                      x_train, y_train,
                                                      x_test, y_test,
                                                      val_x=x_val, val_y=y_val,
                                                      early_stopping_rounds=shared_gbdt_params['early_stopping_rounds'])

        ds_train_val = merge_init_datasets(ds_train, ds_val)
        set_gbtd = eval.train_and_predict_set_gbdt(set_params,
                                                   ds_train_val, np.concatenate([y_train, y_val]),
                                                   ds_test, y_test)

        if args.save:
            pkl_filename = os.path.join(log_dir, '{}_model.pkl'.format(args.exp_name))
            with open(pkl_filename, 'wb') as file:
                pickle.dump(set_gbtd, file)

