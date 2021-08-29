import copy
import numpy as np
import logging
import random
from pprint import pformat

from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble._gb import GradientBoostingClassifier, GradientBoostingRegressor
import xgboost as xgb

from settree.set_data import set_object_to_matrix
from settree.set_tree import SetTree, SetSplitNode
from settree.gbest import GradientBoostedSetTreeClassifier, GradientBoostedSetTreeRegressor
from exps.eval_utils.metrics import acc, mse
from exps.eval_utils.general import load_pickle, Timer



def random_params_search_set_tree(ds_train, train_y, ds_test, test_y,
                                  n_experiments,
                                  params_dict, search_params_dict,
                                  mode='bin_cls'):

    best_test_acc = 0.0
    best_config = None
    best_model_copy = None
    logging.info('Starting random params search for SetTree for {} rounds'.format(n_experiments))

    # np.greater: Return the truth value of (x1 > x2) element-wise.
    condition = np.greater if 'cls' in mode else np.less
    for counter in range(n_experiments):
        params_dict_copy = copy.deepcopy(params_dict)
        for k, v in search_params_dict.items():
            params_dict_copy[k] = random.choice(v)

        logging.info('\nExp[{}]'.format(counter))
        model, train_met, test_met = train_and_predict_set_gbdt(params_dict_copy,
                                                                ds_train, train_y,
                                                                ds_test, test_y,
                                                                mode, out_metrics=True)

        if condition(test_met, best_test_acc):
            best_config = copy.deepcopy(params_dict_copy)
            best_test_acc = copy.deepcopy(test_met)
            best_model_copy = copy.deepcopy(model)

    logging.info('##################################################')
    logging.info('Best experiment test metric: {}'.format(best_test_acc))
    logging.info(pformat(best_config))
    return best_model_copy, best_config, best_test_acc


def random_params_search(classifier,
                         train_x, train_y, test_x, test_y,
                         n_experiments,
                         params_dict, search_params_dict,
                         val_x=None, val_y=None,
                         early_stopping_rounds=5,
                         mode='bin_cls'):

    best_test_acc = 0.0
    best_config = None
    logging.info('Starting random params search for {} for {} rounds'.format(classifier,
                                                                             n_experiments))
    # to use early stopping in sklearn framework
    if classifier == 'sklearn' and early_stopping_rounds != None:
        n_iter_no_change = early_stopping_rounds
        params_dict['n_iter_no_change'] = n_iter_no_change

    for counter in range(n_experiments):
        params_dict_copy = copy.deepcopy(params_dict)
        for k, v in search_params_dict.items():
            params_dict_copy[k] = random.choice(v)

        logging.info('\nExp[{}]'.format(counter))
        if classifier == 'xgboost':
            _, train_acc, test_acc = train_and_predict_xgboost(params_dict_copy,
                                                               train_x, train_y,
                                                               test_x, test_y,
                                                               val_x, val_y,
                                                               early_stopping_rounds,
                                                               mode,
                                                               out_metrics=True)
        elif classifier == 'sklearn':
            _, train_acc, test_acc = train_and_predict_sklearn_gbtd(params_dict_copy,
                                                               train_x, train_y,
                                                               test_x, test_y,
                                                               mode)
        else:
            raise ValueError('Invalid classifier {}'.format(classifier))

        if test_acc > best_test_acc:
            best_config = params_dict_copy
            best_test_acc = test_acc

    logging.info('##################################################')
    logging.info('Best experiment test metric: {}'.format(best_test_acc))
    logging.info(pformat(best_config))
    return best_config, best_test_acc


def split_to_random_sets(x, min_size=2, max_size=20):
    '''

    Parameters
    ----------
    x : <numpy.ndarray> input data shape (N, d)
    min_size : int
    max_size  : int
    Returns
    -------
    list of <numpy.ndarray>
    '''
    if not(isinstance(x, np.ndarray)):
        x = np.array(x)

    n_items = len(x)
    sizes = []
    while(True):
        sizes.append(random.choice(range(min_size, max_size)))
        if sum(sizes) > n_items:
            break
    sizes = np.cumsum(np.array(sizes))
    if sizes[-1] >= n_items:
        sizes = sizes[:-1]
    return np.split(x, indices_or_sections=sizes, axis=0)


def eval_sklearn_dt(eval_dt, set_dataset_train, set_dataset_test, verbose=True):
    sklearn_dt = DecisionTreeClassifier(criterion="entropy")

    sk_train_x, sk_train_y = set_object_to_matrix(set_dataset_train, eval_dt.splitter.operations)
    sk_test_x, sk_test_y = set_object_to_matrix(set_dataset_test, eval_dt.splitter.operations)
    sklearn_dt = sklearn_dt.fit(sk_train_x, sk_train_y)

    if verbose:
        sklearn_train_acc = (sklearn_dt.predict(sk_train_x) == sk_train_y).mean()
        sklearn_test_acc = (sklearn_dt.predict(sk_test_x) == sk_test_y).mean()

        train_acc = (eval_dt.predict(set_dataset_train) == set_dataset_train.y).mean()
        test_acc = (eval_dt.predict(set_dataset_test) == set_dataset_test.y).mean()

        print('SklearnTree: train acc {:.4f} | test acc : {:.4f}'.format(sklearn_train_acc, sklearn_test_acc))
        print('SetTree: train acc {:.4f} | test acc : {:.4f}'.format(train_acc, test_acc))
    return sklearn_dt


def train_decision_tree(ds_train, y_train, ds_test, y_test,
                        splitter, use_attention_set, use_attention_set_comp, attention_set_limit, tree_args):
    ''' Train a single DT and compare to Sklearn'''

    dt = SetTree(attention_set_limit=attention_set_limit,
                 use_attention_set=use_attention_set,
                 use_attention_set_comp=use_attention_set_comp,
                 splitter=splitter,
                 **tree_args)

    logging.info('############ Set tree ############ ')
    timer = Timer()
    dt.fit(ds_train, y_train)
    logging.info('Train took: {}'.format(timer.end()))

    timer = Timer()
    train_preds = dt.predict(ds_train)
    logging.info('Eval train took: {}'.format(timer.end()))
    test_preds = dt.predict(ds_test)

    train_acc = (train_preds == y_train).mean()
    test_acc = (test_preds == y_test).mean()

    logging.info('Results : train acc {:.4f} | test acc : {:.4f}'.format(train_acc, test_acc))
    logging.info('Tree depth: {} n_leafs: {}'.format(dt.depth, dt.n_leafs))

    operations = getattr(dt, 'operations', False) if getattr(dt, 'operations', False) else dt.splitter.operations
    sk_train_x = set_object_to_matrix(ds_train, operations)
    sk_test_x = set_object_to_matrix(ds_test, operations)

    sklearn_dt = DecisionTreeClassifier(criterion="entropy")
    logging.info('############ Sklearn ############ ')
    timer = Timer()
    sklearn_dt = sklearn_dt.fit(sk_train_x, y_train)
    logging.info('Train took: {}'.format(timer.end()))

    timer = Timer()
    sklearn_train_preds = sklearn_dt.predict(sk_train_x)
    logging.info('Eval train took: {}'.format(timer.end()))

    sklearn_train_acc = (sklearn_train_preds == y_train).mean()
    sklearn_test_acc = (sklearn_dt.predict(sk_test_x) == y_test).mean()

    logging.info('Results : train acc {:.4f} | test acc : {:.4f}'.format(sklearn_train_acc, sklearn_test_acc))
    logging.info('Tree depth: {} n_leafs: {}'.format(sklearn_dt.tree_.max_depth, sklearn_dt.tree_.node_count))
    return dt, sklearn_dt


def count_parametres(gb):

    N_PARAMS_NODE = 5
    N_PARAMS_LEAF = 1

    def count_nodes(node, count=0):
        if isinstance(node, SetSplitNode):
            return 1 + count_nodes(node.right, count) + count_nodes(node.left, count)
        else:
            return 0

    count = 0
    for tree in gb.estimators_.flatten():
        count += count_nodes(tree.tree_, count=0) * N_PARAMS_NODE
        count += tree.n_leafs * N_PARAMS_LEAF
    return count


def load_checkpoint_gbdt(checkpoint):
    gbdt = load_pickle(checkpoint)

    none_estimators_inds = np.where(gbdt.estimators_[:, 0] == None)[0]
    if hasattr(gbdt, 'n_estimators_'):
        n_stages = gbdt.n_estimators_

    elif len(none_estimators_inds):
        n_stages = min(none_estimators_inds)

    else:
        n_stages = gbdt.n_estimators

    if n_stages < gbdt.n_estimators:
        gbdt.estimators_ = gbdt.estimators_[:n_stages]
        gbdt.train_score_ = gbdt.train_score_[:n_stages]
        if hasattr(gbdt, 'oob_improvement_'):
            gbdt.oob_improvement_ = gbdt.oob_improvement_[:n_stages]
    return gbdt


def train_and_predict_set_gbdt(params, ds_train, train_y, ds_test, test_y,
                               mode='bin_cls', out_metrics=False, resume=None, eval_train=True, verbose=True):
    # mode : bin_cls, multi_cls, reg
    if verbose:
        logging.info('############ Set GBDT ############ ')
        logging.info('Params:\n' + pformat(params))

    if mode == 'bin_cls':
        gbdt = GradientBoostedSetTreeClassifier(**params)
        eval_met = acc
        eval_met_name = 'acc'

    elif mode == 'multi_cls':
        gbdt = GradientBoostedSetTreeClassifier(**params)
        eval_met = acc
        eval_met_name = 'acc'
    else:
        gbdt = GradientBoostedSetTreeRegressor(**params)
        eval_met = mse
        eval_met_name = 'mse'

    timer = Timer()

    if resume != None:
        gbdt = load_pickle(resume)

        # if it is a checkpoint - saved before completed the train - resize the estimators_ array
        none_estimators_inds = np.where(gbdt.estimators_[:, 0] == None)[0]
        if hasattr(gbdt, 'n_estimators_'):
            n_stages = gbdt.n_estimators_

        elif len(none_estimators_inds):
            n_stages = min(none_estimators_inds)

        else:
            n_stages = gbdt.n_estimators

        if n_stages < gbdt.n_estimators:
            gbdt.estimators_ = gbdt.estimators_[:n_stages]
            gbdt.train_score_ = gbdt.train_score_[:n_stages]
            if hasattr(gbdt, 'oob_improvement_'):
                gbdt.oob_improvement_ = gbdt.oob_improvement_[:n_stages]

        logging.info('Loaded model from {}, with {} trees, resume training'.format(resume, n_stages))

        gbdt.set_params(**{'n_estimators': n_stages + params['n_estimators']})
        logging.info('Continue training for {} estimators'.format(params['n_estimators']))

        logging.info('Warning: continue training with the previous parameters')
        logging.info('Original model parameters:')
        logging.info(pformat(params))

    gbdt.fit(ds_train, train_y)

    if verbose:
        logging.info('Train took: {}'.format(timer.end()))

    if mode == 'bin_cls':
        timer = Timer()
        if eval_train:
            train_raw_predictions = gbdt.decision_function(ds_train)
            if verbose:
                logging.info('Eval train took: {}'.format(timer.end()))
        else:
            logging.info('Skipped train evaluation - train metrics are irrelevant')
            train_raw_predictions = np.zeros((len(ds_train),)) # tmp solution

        test_raw_predictions = gbdt.decision_function(ds_test)
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
        if verbose:
            logging.info('Results : train {} {:.6f} auc: {:.6f} | test {} : {:.4f} auc: {:.4f}'.format(eval_met_name, train_met,
                                                                                                      train_auc, eval_met_name,
                                                                                                      test_met, test_auc))
    else:
        timer = Timer()
        if eval_train:
            train_preds = gbdt.predict(ds_train)
            if verbose:
                logging.info('Eval train took: {}'.format(timer.end()))
        else:
            logging.info('Skipped train evaluation - train metrics are irrelevant')
            train_preds = np.zeros((len(ds_train),)) # tmp solution

        test_preds = gbdt.predict(ds_test)
        train_met = eval_met(train_y, train_preds)
        test_met = eval_met(test_y, test_preds)
        if verbose:
            logging.info('Results : train {} {:.6f} | test {} : {:.6f}'.format(eval_met_name, train_met,
                                                                               eval_met_name, test_met))

    depths = []
    n_leafs = []
    n_stages, K = gbdt.estimators_.shape
    for i in range(n_stages):
        for k in range(K):
            depths.append(gbdt.estimators_[i, k].depth)
            n_leafs.append(gbdt.estimators_[i, k].n_leafs)

    depths = np.array(depths)
    n_leafs = np.array(n_leafs)
    if verbose:
        logging.info('Trees sizes stats: depth: {:.1f}+-{:.3f} | n_leafs: {:.1f}+-{:.3f}'.format(depths.mean(), depths.std(),
                                                                                                 n_leafs.mean(), n_leafs.std()))
    if out_metrics:
        return gbdt, train_met, test_met
    else:
        return gbdt


def train_and_predict_set_tree(params, ds_train, train_y, ds_test, test_y,
                               mode='bin_cls', out_metrics=False, verbose=True):

    # mode : bin_cls, multi_cls, reg
    if verbose:
        logging.info('############ Set Tree ############ ')
        logging.info('Params:\n' + pformat(params))

    tree = SetTree(**params)

    if mode == 'bin_cls':
        eval_met = acc
        eval_met_name = 'acc'

    elif mode == 'multi_cls':
        eval_met = acc
        eval_met_name = 'acc'
    else:
        eval_met = mse
        eval_met_name = 'mse'

    timer = Timer()
    tree.fit(ds_train, train_y)
    if verbose:
        logging.info('Train took: {}'.format(timer.end()))

    if mode == 'bin_cls':
        timer = Timer()
        train_probs = tree.predict_proba(ds_train)
        train_preds = tree.predict(ds_train)
        if verbose:
            logging.info('Eval train took: {}'.format(timer.end()))

        test_probs = tree.predict_proba(ds_test)
        test_preds = tree.predict(ds_test)

        train_met = eval_met(train_y, train_preds)
        test_met = eval_met(test_y, test_preds)

        train_auc = roc_auc_score(train_y, train_probs[:, 1])
        test_auc = roc_auc_score(test_y, test_probs[:, 1])
        if verbose:
            logging.info('Results : train {} {:.6f} auc: {:.6f} | test {} : {:.4f} auc: {:.4f}'.format(eval_met_name, train_met,
                                                                                                      train_auc, eval_met_name,
                                                                                                      test_met, test_auc))
    else:
        timer = Timer()
        train_preds = tree.predict(ds_train)
        if verbose:
            logging.info('Eval train took: {}'.format(timer.end()))
        test_preds = tree.predict(ds_test)
        train_met = eval_met(train_y, train_preds)
        test_met = eval_met(test_y, test_preds)
        if verbose:
            logging.info('Results : train {} {:.6f} | test {} : {:.6f}'.format(eval_met_name, train_met,
                                                                               eval_met_name, test_met))
    if out_metrics:
        return tree, train_met, test_met
    else:
        return tree


def train_and_predict_xgboost(params,
                              train_x, train_y, test_x, test_y, val_x=None, val_y=None,
                              early_stopping_rounds=None, mode='bin_cls', out_metrics=False, verbose=True):
    if verbose:
        logging.info('############ XGBoost ############ ')
        logging.info('Params:\n' + pformat(params))

    if mode == 'bin_cls':
        gbdt = xgb.XGBClassifier(**params)
        eval_met = acc
        eval_met_name = 'acc'

    elif mode == 'multi_cls':
        gbdt = xgb.XGBClassifier(**params)
        eval_met = acc
        eval_met_name = 'acc'
    else:
        gbdt = xgb.XGBRegressor(**params)
        eval_met = mse
        eval_met_name = 'mse'

    if verbose:
        logging.info('Params: {}'.format(params))
    timer = Timer()
    if np.any(val_x):
        gbdt.fit(X=train_x,
                y=train_y,
                eval_set=[(val_x, val_y)],
                early_stopping_rounds=early_stopping_rounds)
    else:
        gbdt.fit(train_x, train_y)

    if verbose:
        logging.info('Train took: {}'.format(timer.end()))

    timer = Timer()
    train_preds = gbdt.predict(train_x)
    if verbose:
        logging.info('Eval train took: {}'.format(timer.end()))
    test_preds = gbdt.predict(test_x)

    train_met = eval_met(train_y, train_preds)
    test_met = eval_met(test_y, test_preds)

    if mode == 'bin_cls':
        train_proba = gbdt.predict_proba(train_x)[:, 1]
        test_proba = gbdt.predict_proba(test_x)[:, 1]

        train_auc = roc_auc_score(train_y, train_proba)
        test_auc = roc_auc_score(test_y, test_proba)
        if verbose:
            logging.info('Results : train {} {:.6f} auc: {:.4f} | test {} : {:.6f} auc: {:.4f}'.format(eval_met_name, train_met,
                                                                                                      train_auc, eval_met_name,
                                                                                                      test_met, test_auc))
    else:
        if verbose:
            logging.info('Results : train {} {:.6f} | test {} : {:.6f}'.format(eval_met_name, train_met,
                                                                               eval_met_name, test_met))
    if out_metrics:
        return gbdt, train_met, test_met
    else:
        return gbdt


def train_and_predict_sklearn_gbtd(params,
                                   train_x, train_y, test_x, test_y,
                                   mode='bin_cls', out_metrics=False, verbose=True):
    if verbose:
        logging.info('############ Sklearn GBDT ############ ')
        logging.info('Params:\n' + pformat(params))

    if mode == 'bin_cls':
        gbdt = GradientBoostingClassifier(**params)
        eval_met = acc
        eval_met_name = 'acc'

    elif mode == 'multi_cls':
        gbdt = GradientBoostingClassifier(**params)
        eval_met = acc
        eval_met_name = 'acc'
    else:
        gbdt = GradientBoostingRegressor(**params)
        eval_met = mse
        eval_met_name = 'mse'

    if verbose:
        logging.info('Params: {}'.format(params))
    timer = Timer()
    gbdt.fit(train_x, train_y)
    if verbose:
        logging.info('Train took: {}'.format(timer.end()))

    if mode == 'bin_cls':
        timer = Timer()
        train_raw_predictions = gbdt.decision_function(train_x)
        if verbose:
            logging.info('Eval train took: {}'.format(timer.end()))
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
        if verbose:
            logging.info('Results : train {} {:.6f} auc: {:.6f} | test {} : {:.4f} auc: {:.4f}'.format(eval_met_name, train_met,
                                                                                                      train_auc, eval_met_name,
                                                                                                      test_met, test_auc))
    else:
        timer = Timer()
        train_preds = gbdt.predict(train_x)
        if verbose:
            logging.info('Eval train took: {}'.format(timer.end()))
        test_preds = gbdt.predict(test_x)
        train_met = eval_met(train_y, train_preds)
        test_met = eval_met(test_y, test_preds)
        if verbose:
            logging.info('Results : train {} {:.6f} | test {} : {:.6f}'.format(eval_met_name, train_met,
                                                                               eval_met_name, test_met))
    if out_metrics:
        return gbdt, train_met, test_met
    else:
        return gbdt


def train_and_predict_sklearn_dt(params,
                                 train_x, train_y, test_x, test_y,
                                 mode='bin_cls', out_metrics=False, verbose=True):
    if verbose:
        logging.info('############ Sklearn DT ############ ')
        logging.info('Params:\n' + pformat(params))

    if mode == 'bin_cls':
        dt = DecisionTreeClassifier(**params)
        eval_met = acc
        eval_met_name = 'acc'

    elif mode == 'multi_cls':
        dt = DecisionTreeClassifier(**params)
        eval_met = acc
        eval_met_name = 'acc'
    else:
        dt = DecisionTreeRegressor(**params)
        eval_met = mse
        eval_met_name = 'mse'

    if verbose:
        logging.info('Params: {}'.format(params))
    timer = Timer()
    dt.fit(train_x, train_y)
    if verbose:
        logging.info('Train took: {}'.format(timer.end()))

    if mode == 'bin_cls':
        timer = Timer()
        train_preds = dt.predict(train_x)
        if verbose:
            logging.info('Eval train took: {}'.format(timer.end()))
        test_preds = dt.predict(test_x)
        train_met = eval_met(train_y, train_preds)
        test_met = eval_met(test_y, test_preds)

        train_probs = dt.predict_proba(train_x)
        test_probs = dt.predict_proba(test_x)

        train_auc = roc_auc_score(train_y, train_probs[:, 1])
        test_auc = roc_auc_score(test_y, test_probs[:, 1])
        if verbose:
            logging.info('Results : train {} {:.6f} auc: {:.6f} | test {} : {:.4f} auc: {:.4f}'.format(eval_met_name, train_met,
                                                                                                      train_auc, eval_met_name,
                                                                                                      test_met, test_auc))
    else:
        timer = Timer()
        train_preds = dt.predict(train_x)
        if verbose:
            logging.info('Eval train took: {}'.format(timer.end()))
        test_preds = dt.predict(test_x)
        train_met = eval_met(train_y, train_preds)
        test_met = eval_met(test_y, test_preds)
        if verbose:
            logging.info('Results : train {} {:.6f} | test {} : {:.6f}'.format(eval_met_name, train_met,
                                                                               eval_met_name, test_met))
    if out_metrics:
        return dt, train_met, test_met
    else:
        return dt

