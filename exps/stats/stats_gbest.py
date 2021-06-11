import numpy as np
import logging
import os
from pprint import pformat
from sklearn.model_selection import train_test_split

from settree.set_data import OPERATIONS, flatten_datasets
import exps.eval_utils as eval
from exps.synthetic_data import get_data_by_task


params = {
    'exp_name': 'exp_7_multi_seed_real_random',
    'seed': 0,
    'n_train': 10000,
    'n_test': 1000,
    'set_size': 10,
}


if __name__ == '__main__':
    log_dir = os.path.join(os.path.abspath('__file__' + '/../'), 'outputs')
    eval.create_logger(log_dir=log_dir,
                       log_name=params['exp_name'],
                       dump=False)

    logging.info(pformat(params))

    seed2results = {}
    for seed in range(5):
        logging.info('Start seed {}'.format(seed))
        np.random.seed(seed)

        shared_gbdt_params = {'n_estimators': 50,
                              'learning_rate': 0.1,
                              'max_depth': 6,
                              'max_features': None,
                              'subsample': 0.5,
                              'n_iter_no_change': 3,
                              'random_state': seed}

        set_params = {'n_estimators': shared_gbdt_params['n_estimators'],
                      'operations': OPERATIONS,
                      'splitter': 'sklearn',
                      'use_attention_set': True,
                      'use_attention_set_comp': True,
                      'attention_set_limit': 5,
                      'max_depth': shared_gbdt_params['max_depth'],
                      'max_features': shared_gbdt_params['max_features'],
                      'subsample': shared_gbdt_params['subsample'],
                      'random_state': shared_gbdt_params['random_state'],
                      'validation_fraction': 0.1,
                      'tol': 1e-3,
                      'n_iter_no_change': shared_gbdt_params['n_iter_no_change'],
                      'verbose': 9}

        xgboost_params = {
            'max_depth': shared_gbdt_params['max_depth'],
            'n_jobs': 10,
            'learning_rate': shared_gbdt_params['learning_rate'],
            'n_estimators': shared_gbdt_params['n_estimators'],
            'colsample_bytree': shared_gbdt_params['max_features'],
            'subsample': shared_gbdt_params['subsample'],
            'reg_lambda': 0,
            'reg_alpha': 0,
            'verbosity': 0,
            'random_state': 5,
            'seed': shared_gbdt_params['random_state']}

        tasks = ['different_laplace_normal', 'different_mean', 'different_std']
        #tasks = ['different_laplace_normal']
        results = {task_name: {'set': [], 'reg': []} for task_name in tasks}
        x = [4, 10, 20, 30, 40, 50]
        for task_name in tasks:
            logging.info('Task: {}'.format(task_name))
            for set_size in x:
                params['set_size'] = set_size
                ds_train, y_train, ds_test, y_test = get_data_by_task(task_name, params)

                x_train, x_test = flatten_datasets(ds_train, ds_test,
                                                   operations_list=set_params['operations'],
                                                   ds_val=None)

                x_train_, x_val, y_train_, y_val = train_test_split(x_train, y_train,
                                                                  test_size=set_params['validation_fraction'],
                                                                  random_state=shared_gbdt_params['random_state'])

                xgboost_gbtd, train_acc, test_acc = eval.train_and_predict_xgboost(xgboost_params,
                                                              x_train_, y_train_,
                                                              x_test, y_test,
                                                              val_x=x_val, val_y=y_val,
                                                              early_stopping_rounds=shared_gbdt_params['n_iter_no_change'],
                                                              mode='bin_cls',
                                                              out_metrics=True,
                                                              verbose=False)

                set_gbtd, set_train_acc, set_test_acc = eval.train_and_predict_set_gbdt(set_params,
                                                                                        ds_train, y_train,
                                                                                        ds_test, y_test,
                                                                                        mode='bin_cls',
                                                                                        out_metrics=True,
                                                                                        verbose=False)
                results[task_name]['reg'].append(test_acc)
                results[task_name]['set'].append(set_test_acc)
                logging.info('Set size: {} | reg:{:.4f} | set:{:.4f}'.format(set_size, test_acc, set_test_acc))
        logging.info('Finish seed {}'.format(seed))
        seed2results[seed] = results

        # plt.plot(np.arange(len(x)), results[task_name]['reg'], label='reg')
        # plt.plot(np.arange(len(x)), results[task_name]['set'], label='set')
        # plt.ylabel('Acc')
        # plt.xlabel('Set size')
        # plt.xticks(np.arange(len(x)), x)
        # plt.title(task_name)
        # plt.legend()
        # plt.show()

    eval.save_json(seed2results, os.path.join(log_dir, '{}_results_dict.json'.format(params['exp_name'])))

    # eval.save_pickle(xgboost_gbtd, os.path.join(log_dir, params['exp_name'] + '_xbg_model.pkl'))
    # eval.save_pickle(set_gbtd, os.path.join(log_dir, params['exp_name'] + '_set_model.pkl'))
