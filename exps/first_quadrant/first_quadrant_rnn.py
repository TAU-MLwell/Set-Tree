from exps.deep_learning import *
from exps.synthetic_data import get_first_quarter_data
import exps.eval_utils as eval


if __name__ == '__main__':

    params = {
        'exp_name': 'rnn_sire_100dim_multi',
        'seed': 0,
        'n_train': 100000,
        'n_test': 10000,
        'train_set_size': 20,
        'set_sizes': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300],
        'dim': 100,

        'bidirectional': False,
        'janossy': False,
        'sire': True,
        'sire_alpha': 0.1,

        'batch_size_train': 64,
        'batch_size_test': 128,
        'device': 'cuda:0',
        'n_epochs': 30,
        'patience': 5,
        'n_workers': 4,
        'n_seeds': 5,
    }

    log_dir = os.path.join(os.path.abspath('__file__' + '/../'), 'outputs', 'SIRE')
    eval.create_logger(log_dir=log_dir,
                       log_name=params['exp_name'],
                       dump=True)

    exp2results = {}
    for seed in range(params['n_seeds']):
        logging.info('Start exp {}'.format(seed))
        set_initial_random_seed(seed)
        exp_accs = {}
        logging.info('Start train: set_size={} dim={}'.format(params['train_set_size'], params['dim']))
        train_data, y_train = get_first_quarter_data(num_samples=params['n_train'],
                                                     min_items_set=params['train_set_size'],
                                                     max_items_set=params['train_set_size'] + 1,
                                                     dim=params['dim'])
        test_data, y_test = get_first_quarter_data(num_samples=params['n_test'],
                                                   min_items_set=params['train_set_size'],
                                                   max_items_set=params['train_set_size'] + 1,
                                                   dim=params['dim'])
        logging.info('Created dataset')

        train_loader = to_dataloader(permute_set(train_data), y_train, DummyDataset, params['batch_size_train'],
                                     num_workers=params['n_workers'], shuffle=True, transform_x=lambda x: x)
        test_loader = to_dataloader(permute_set(test_data), y_test, DummyDataset, params['batch_size_test'],
                                    num_workers=params['n_workers'], shuffle=False, transform_x=lambda x: x)
        logging.info('Finish preprocessing')

        model = RNNModel(cell_type=nn.LSTM,
                         project_in=nn.Linear(params['dim'], 64),
                         project_out=nn.Linear(64, 2),
                         input_dim=64,
                         hidden_dim=64,
                         dropout=0.2,
                         num_layers=1,
                         bidirectional=params['bidirectional'])

        model = model.to(params['device'])
        logging.info('Number of parametres {}'.format(count_parameters(model)))
        criteria = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        trainer = RegularTrainer(n_epochs=params['n_epochs'],
                                 criteria=criteria,
                                 optimizer=optimizer,
                                 eval_metric=eval.AverageAcc,
                                 janossy=params['janossy'],
                                 sire=params['sire'],
                                 sire_alpha=params['sire_alpha'],
                                 device=params['device'],
                                 early_stop=EarlyStopping(metric_name='loss',
                                                          patience=params['patience'],
                                                          min_is_better=False), verbose=3)

        trainer.fit(train_loader, test_loader, model)

        exp_accs = []
        for test_set_size in params['set_sizes']:
            test_data, y_test = get_first_quarter_data(num_samples=params['n_test'],
                                                       min_items_set=test_set_size,
                                                       max_items_set=test_set_size + 1,
                                                       dim=params['dim'])
            test_loader = to_dataloader(permute_set(test_data), y_test, DummyDataset, params['batch_size_test'],
                                        num_workers=0, shuffle=False, transform_x=lambda x: x)
            if params['janossy']:
                test_loss, test_acc = trainer.test_janossy(model, test_loader, k=params['train_set_size'], metric=None)
            else:
                test_loss, test_acc = trainer.test(trainer.best_model, test_loader)
            logging.info('Test set size: {} | test loss: {:.5f} test acc: {:.5f}'.format(test_set_size, test_loss, test_acc))
            exp_accs.append(test_acc)
        exp2results[seed] = exp_accs

    eval.save_json(exp2results, os.path.join(log_dir, '{}_results_dump.json'.format(params['exp_name'])))
