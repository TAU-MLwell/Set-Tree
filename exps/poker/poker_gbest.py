import numpy as np
import os
import logging
from pprint import pformat

from settree.set_data import SetDataset, OPERATIONS, flatten_datasets
from exps.data import get_poker_dataset
import exps.eval_utils as eval


params = {
    'exp_name': 'single_new_ops',
    'seed': 0,
    'n_train': 25000,
    'n_test': 10000,
    'n_exp': 5
}


if __name__ == '__main__':

    log_dir = os.path.join(os.path.abspath('__file__' + '/../'), 'outputs', 'single')
    eval.create_logger(log_dir=log_dir,
                       log_name=params['exp_name'],
                       dump=True)

    logging.info(pformat(params))
    logging.info(pformat(params))

    mets = {'reg': [], 'reg_proc': [], 'set': []}
    for seed in range(params['n_exp']):
        np.random.seed(seed)
        logging.info('####### Exp {} #######'.format(seed))
        x_train, y_train, x_test, y_test = get_poker_dataset(pre_process_method='set_tree',
                                                             scenerio='bin',
                                                             n_test=params['n_test'],
                                                             seed=seed)
        # mode = 'bin_cls'
        mode = 'multi_cls'
        logging.info('There are {} train and {} test'.format(len(x_train), len(x_test)))

        ds_train = SetDataset(records=x_train, is_init=True)
        ds_test = SetDataset(records=x_test, is_init=True)

        shared_gbdt_params = {'n_estimators': 150,
                              'learning_rate': 0.1,
                              'max_depth': 6,
                              'max_features': None,
                              'subsample': 1,
                              'n_iter_no_change': 3,
                              'random_state': 42}

        logging.info('Shared params:\n' + pformat(shared_gbdt_params))

        set_params = {'n_estimators': shared_gbdt_params['n_estimators'],
                      'operations': OPERATIONS,
                      'splitter': 'sklearn',
                      'use_attention_set': True,
                      'attention_set_limit': 5,
                      'max_depth': shared_gbdt_params['max_depth'],
                      'max_features': shared_gbdt_params['max_features'],
                      'subsample': shared_gbdt_params['subsample'],
                      'random_state': shared_gbdt_params['random_state'],
                      # 'validation_fraction': 0.1,
                      # 'tol': 1e-3,
                      # 'n_iter_no_change': shared_gbdt_params['n_iter_no_change'],
                      'verbose': 3}

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

        ################################################################################################################
        xgboost_gbtd_proc, reg_proc_train_cc, reg_proc_test_acc = eval.train_and_predict_xgboost(xgboost_params,
                                                      x_train, y_train,
                                                      x_test, y_test,
                                                      val_x=None, val_y=None,
                                                      early_stopping_rounds=None,
                                                      out_metrics=True,
                                                      mode=mode)
        mets['reg_proc'].append(reg_proc_test_acc)
        ################################################################################################################

        xgboost_params = {  # 'objective': 'binary:logistic', # 'multi:softmax', binary:logistic
            'max_depth': shared_gbdt_params['max_depth'],
            'n_jobs': 10,
            'learning_rate': shared_gbdt_params['learning_rate'],
            'n_estimators': 300,
            'colsample_bytree': shared_gbdt_params['max_features'],
            'subsample': shared_gbdt_params['subsample'],
            'reg_lambda': 0,
            'reg_alpha': 0,
            'verbosity': 0,
            'random_state': shared_gbdt_params['random_state'],
            'seed': shared_gbdt_params['random_state']}
        xgboost_gbtd, reg_train_cc, reg_test_acc = eval.train_and_predict_xgboost(xgboost_params,
                                                      np.stack([r.flatten() for r in ds_train.records]), y_train,
                                                      np.stack([r.flatten() for r in ds_test.records]), y_test,
                                                      val_x=None, val_y=None,
                                                      early_stopping_rounds=None,
                                                      out_metrics=True,
                                                      mode=mode)
        mets['reg'].append(reg_test_acc)

        ################################################################################################################
        set_gbtd, set_train_cc, set_test_acc = eval.train_and_predict_set_gbdt(set_params,
                                                                               ds_train, y_train,
                                                                               ds_test, y_test,
                                                                               out_metrics=True,
                                                                               mode=mode)
        mets['set'].append(set_test_acc)
        if seed == 0:
            eval.save_pickle(set_gbtd, os.path.join(log_dir, '{}_set_model.pickle'.format(params['exp_name'])))

        ################################################################################################################
        logging.info('Seed: {} reg_proc acc: {:.5f} | reg acc: {:.5f} | set acc {:.5f}'.format(seed,
                                                                                               reg_proc_test_acc,
                                                                                               reg_test_acc,
                                                                                               set_test_acc))
    ################################################################################################################
    # End of train
    ################################################################################################################
    reg_proc_res = np.array(mets['reg_proc'])
    logging.info('Reg-Proc: mean: {:.4f} | std: {:.4f}'.format(reg_proc_res.mean(), reg_proc_res.std()))
    reg_res = np.array(mets['reg'])
    logging.info('Reg: mean: {:.4f} | std: {:.4f}'.format(reg_res.mean(), reg_res.std()))
    set_res = np.array(mets['set'])
    logging.info('Set: mean: {:.4f} | std: {:.4f}'.format(set_res.mean(), set_res.std()))

    # eval.save_pickle(xgboost_gbtd, os.path.join(os.path.join(log_dir, params['exp_name'] + '_xbg_model.pkl')))
    # eval.save_pickle(set_gbtd, os.path.join(os.path.join(log_dir, params['exp_name'] + '_set_model.pkl')))

    ####################################################################################################################
    # ablation: train tree with permutations data
    ####################################################################################################################
    # x_test_numpy, y_test_numpy = pre_process_numpy(test[:10000])
    # x_perm_train, y_perm_train = pre_process_all_permutations(train)
    # logging.info('Flatten labels')
    # y_test_numpy[y_test_numpy != 0] = 1
    # y_perm_train[y_perm_train != 0] = 1
    # mode = 'bin_cls'
    #
    # x_perm_train, x_perm_val, y_perm_train, y_perm_val = train_test_split(x_perm_train, y_perm_train,
    #                                                                       test_size=0.15,
    #                                                                       random_state=params['seed'])
    #
    # xgboost_params = {'tree_method': 'gpu_hist',
    #                   'max_depth': 12,
    #                   'n_jobs': 10,
    #                   'learning_rate': 0.05,
    #                   'n_estimators': 500,
    #                   'colsample_bytree': None,
    #                   'subsample': 0.5,
    #                   'reg_lambda': 0,
    #                   'reg_alpha': 0,
    #                   'verbosity': 3}
    #
    # xgboost_gbtd = eval.train_and_predict_xgboost(xgboost_params,
    #                                               x_perm_train, y_perm_train,
    #                                               x_test_numpy, y_test_numpy,
    #                                               val_x=x_perm_val, val_y=y_perm_val,
    #                                               early_stopping_rounds=10,
    #                                               mode=mode)
    #
