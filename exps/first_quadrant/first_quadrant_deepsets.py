from exps.deep_learning import *
from exps.synthetic_data import get_first_quarter_data
import exps.eval_utils as eval


if __name__ == '__main__':
    params = {'exp_name': '2dim_multi_dss_max', # '2dim_multi_dss_sum',
              'seed': 0,
              'n_train': 100000,
              'n_test': 10000,
              'dim': 2,
              'train_set_size': 20,
              'test_sizes': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300],
              'n_exp': 5,

              'batch_size_train': 128,
              'batch_size_test': 256,
              'num_workers': 4,
              'device': 'cuda:4',
              'n_epochs': 50
              }

    log_dir = os.path.join(os.path.abspath('__file__' + '/../'), 'outputs', 'DSS')
    eval.create_logger(log_dir=log_dir,
                       log_name=params['exp_name'],
                       dump=True)

    exp2results = {}
    for seed in range(params['n_exp']):
        logging.info('Start exp {}'.format(seed))
        params['seed'] = seed
        np.random.seed(seed)

        logging.info('Start train: set_size={} dim={}'.format(params['n_train'], params['dim']))
        train_data, y_train = get_first_quarter_data(num_samples=params['n_train'],
                                                     min_items_set=params['train_set_size'],
                                                     max_items_set=params['train_set_size'] + 1,
                                                     dim=params['dim'])
        train_loader = to_dataloader(permute_set(train_data), y_train, DummyDataset, params['batch_size_train'],
                                     num_workers=params['num_workers'], shuffle=True, transform_x=lambda x: x)
        test_data, y_test = get_first_quarter_data(num_samples=params['n_test'],
                                                     min_items_set=params['train_set_size'],
                                                     max_items_set=params['train_set_size'] + 1,
                                                     dim=params['dim'])
        test_loader = to_dataloader(permute_set(test_data), y_test, DummyDataset, params['batch_size_test'],
                                     num_workers=params['num_workers'], shuffle=True, transform_x=lambda x: x)

        logging.info('Finish preprocessing')

        model = DSSInvarianceModel([params['dim'], 64, 128, 128],
                                   drop_rate=0.2,
                                   rho=nn.Sequential(nn.Linear(128, 2)),
                                   op=protected_max)

        # model = InvarianceModel(theta=nn.Sequential(nn.Linear(params['dim'], 64),
        #                                             nn.ReLU(),
        #                                             nn.Dropout(0.2),
        #                                             nn.Linear(64, 128),
        #                                             nn.ReLU(),
        #                                             nn.Dropout(0.2),
        #                                             nn.Linear(128, 128)),
        #                         rho=nn.Sequential(nn.Linear(128, 2)),
        #                         op=torch.sum)
        logging.info(model)
        logging.info('Number of parametres {}'.format(count_parameters(model)))
        model = model.to(params['device'])
        criteria = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        trainer = RegularTrainer(n_epochs=params['n_epochs'],
                                 criteria=criteria,
                                 optimizer=optimizer,
                                 eval_metric=eval.AverageAcc,
                                 device=params['device'],
                                 early_stop=EarlyStopping(metric_name='loss',
                                                      patience=8,
                                                      min_is_better=True), verbose=3)
        trainer.fit(train_loader, test_loader, model)


        exp_accs = []
        for test_set_size in params['test_sizes']:
            test_data, y_test = get_first_quarter_data(num_samples=params['n_test'],
                                                       min_items_set=test_set_size,
                                                       max_items_set=test_set_size + 1,
                                                       dim=params['dim'])
            test_loader = to_dataloader(permute_set(test_data), y_test, DummyDataset, params['batch_size_test'],
                                        num_workers=0, shuffle=False, transform_x=lambda x: x)

            test_loss, test_acc = trainer.test(trainer.best_model, test_loader)
            logging.info('Test set size: {} | test loss: {:.5f} test acc: {:.5f}'.format(test_set_size, test_loss, test_acc))
            exp_accs.append(test_acc)
        exp2results[seed] = exp_accs

    eval.save_json(exp2results, os.path.join(log_dir, '{}_results_dump.json'.format(params['exp_name'])))
