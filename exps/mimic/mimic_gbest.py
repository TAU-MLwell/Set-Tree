import numpy as np
import logging
import os

from settree.set_data import OPERATIONS, flatten_datasets
import exps.eval_utils as eval
from exps.data import get_mimic_data_fold

if __name__ == '__main__':
    params = {'exp_name': 'mild_multi_multiple_seeds',
              'seed': 0,
              'type': 'mild',
              'num': 1}

    log_dir = os.path.join(os.path.abspath('__file__' + '/../'), 'outputs', 'mild')
    eval.create_logger(log_dir=log_dir,
                       log_name=params['exp_name'],
                       dump=True)

    mets = {'reg': [], 'set': []}
    for seed in range(5):
        logging.info('Seed {}'.format(seed))
        dataset = get_mimic_data_fold(params['type'], seed)

        ds_train = dataset['ds_train']
        ds_test = dataset['ds_test']
        y_train = dataset['y_train']
        y_test = dataset['y_test']

        shared_gbdt_params = {'n_estimators': 170,
                              'learning_rate': 0.1,
                              'max_depth': 5,
                              'max_features': None,
                              'subsample': 0.5,
                              'random_state': seed}

        set_params = {'n_estimators': shared_gbdt_params['n_estimators'],
                      'operations':  OPERATIONS,
                      'splitter': 'sklearn',
                      'use_attention_set': True,
                      'attention_set_limit': 4,
                      'max_depth': shared_gbdt_params['max_depth'],
                      'max_features': shared_gbdt_params['max_features'],
                      'subsample': shared_gbdt_params['subsample'],
                      'random_state': shared_gbdt_params['random_state'],
                      'verbose': 3}

        sklearn_params = {'n_estimators': shared_gbdt_params['n_estimators'],
                          'criterion': 'mse',
                          'learning_rate': shared_gbdt_params['learning_rate'],
                          'max_depth': shared_gbdt_params['max_depth'],
                          'max_features': shared_gbdt_params['max_features'],
                          'subsample': shared_gbdt_params['subsample'],
                          'random_state': shared_gbdt_params['random_state']}

        xgboost_params = {  # 'objective': 'binary:logistic', # 'multi:softmax', binary:logistic
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

        reg_gbdt, reg_train_met, reg_test_met = eval.train_and_predict_xgboost(xgboost_params,
                                                      x_train, y_train,
                                                      x_test, y_test,
                                                      val_x=None, val_y=None,
                                                      out_metrics=True,
                                                      early_stopping_rounds=None,
                                                      mode='bin_cls')

        set_gbdt, set_train_met, set_test_met = eval.train_and_predict_set_gbdt(set_params,
                                                                                ds_train, y_train,
                                                                                ds_test, y_test,
                                                                                out_metrics=True,
                                                                                mode='bin_cls')
        mets['reg'].append(reg_test_met)
        mets['set'].append(set_test_met)

    eval.save_json(mets, os.path.join('{}_multi_results.json'.format(params['exp_name'])))
    reg_res = np.array(mets['reg'])
    logging.info('Reg: mean: {:.4f} | std: {:.4f}'.format(reg_res.mean(), reg_res.std()))
    set_res = np.array(mets['set'])
    logging.info('Set: mean: {:.4f} | std: {:.4f}'.format(set_res.mean(), set_res.std()))
