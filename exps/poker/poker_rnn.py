from pprint import pformat
from exps.data import get_poker_dataset
from exps.deep_learning import *
import exps.eval_utils as eval

params = {
    'exp_name': 'sire_cls',

    'seed': 0,
    'n_train': 25000,
    'n_test': 10000,

    'pre_process_method': 'deepsets',
    'scenerio': 'bin',

    'janossy': False,
    'sire': True,
    'sire_alpha': 0.001,

    'input_dim': 2,
    'output_dim': 2,
    'dropout': 0.2,

    'batch_size_train': 128,
    'batch_size_test': 128,

    'lr': 1e-3,
    'weight_decay': 0.0,

    'device': 'cuda:4',
    'n_epochs': 100,
    'patience': 5,
    'n_workers': 4,
    'n_exps': 5
}

if __name__ == '__main__':

    log_dir = os.path.join(os.path.abspath('__file__' + '/../'), 'outputs', 'SIRE')
    eval.create_logger(log_dir=log_dir,
                       log_name=params['exp_name'],
                       dump=True)

    logging.info(pformat(params))
    np.random.seed(params['seed'])

    x_train, y_train, x_test, y_test = get_poker_dataset(pre_process_method=params['pre_process_method'], # deepsets
                                                         scenerio=params['scenerio'],
                                                         n_test=10000,
                                                         seed=params['seed'])

    logging.info('Created dataset')
    train_loader = to_dataloader(x_train, y_train, DummyDataset, params['batch_size_train'],
                                 num_workers=params['n_workers'],
                                 shuffle=True,
                                 transform_x=lambda x: x,
                                 collate_fn=collate_func_rnn)
    test_loader = to_dataloader(x_test, y_test, DummyDataset, params['batch_size_test'],
                                num_workers=params['n_workers'],
                                shuffle=False,
                                transform_x=lambda x: x,
                                collate_fn=collate_func_rnn)

    logging.info('Finish preprocessing')
    logging.info('Num train {} num test {}'.format(len(train_loader.dataset), len(test_loader.dataset)))

    # Big DeepSets
    mets = []
    for i in range(params['n_exps']):
        set_initial_random_seed(i)

        model = RNNModel(cell_type=nn.LSTM,
                         project_in=nn.Linear(params['input_dim'], 64),
                         project_out=nn.Linear(64, params['output_dim']),
                         input_dim=64,
                         hidden_dim=64,
                         dropout=params['dropout'],
                         num_layers=1,
                         bidirectional=False)

        logging.info('There are {} params'.format(count_parameters(model)))
        logging.info(str(model))
        model = model.to(params['device'])

        criteria = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        # optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=0.9)

        trainer = RegularTrainer(n_epochs=params['n_epochs'],
                                 criteria=criteria,
                                 optimizer=optimizer,
                                 eval_metric=eval.AverageAcc,
                                 device=params['device'],
                                 janossy=params['janossy'],
                                 sire=params['sire'],
                                 sire_alpha=params['sire_alpha'],
                                 early_stop=EarlyStopping(metric_name='loss',
                                                          patience=params['patience'],
                                                          min_is_better=True), verbose=3)

        trainer.fit(train_loader, test_loader, model)

        # test the full test_dataset
        _, _, x_test_full, y_test_full = get_poker_dataset(pre_process_method=params['pre_process_method'],
                                                             scenerio=params['scenerio'],
                                                             n_test=None,
                                                             seed=params['seed'])
        test_full_loader = to_dataloader(x_test_full, y_test_full,
                                         DummyDataset, params['batch_size_test'],
                                         num_workers=params['n_workers'],
                                         shuffle=False,
                                         transform_x=lambda x: x)
        test_loss, test_acc = trainer.test(model, test_full_loader, metric=None)
        mets.append(test_acc)
    mets = np.array(mets)
    logging.info('Finish: mean acc:{:.4f} std: {:.4f}'.format(mets.mean(), mets.std()))