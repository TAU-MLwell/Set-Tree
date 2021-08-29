import json
from pprint import pformat
from exps.deep_learning import *
import exps.eval_utils as eval
from exps.synthetic_data import get_data_by_task

params = {
    'exp_name': 'sire',
    'seed': 0,
    'n_train': 50000,
    'n_test': 5000,

    'janossy': False,
    'sire': True,
    'sire_alpha': 0.01,

    'set_size': 10,
    'batch_size_train': 128,
    'batch_size_test': 128,

    'n_workers': 10,
    'lr': 1e-3,
    'weight_decay': 0.0,
    'device': 'cuda:1',
    'n_epochs': 50,
    'patience': 5
}

if __name__ == '__main__':
    log_dir = os.path.join(os.path.abspath('__file__' + '/../'), 'outputs', 'SIRE')
    eval.create_logger(log_dir=log_dir,
                       log_name=params['exp_name'],
                       dump=True)

    logging.info(pformat(params))

    for seed in range(5):
        logging.info('Start seed {}'.format(seed))
        np.random.seed(seed)

        tasks = ['different_laplace_normal', 'different_mean', 'different_std']
        x = [4, 10, 20, 30, 40, 50]
        results = {task_name: {i: [] for i in x} for task_name in tasks}
        for task_name in tasks:
            logging.info('Task: {}'.format(task_name))
            for set_size in x:
                logging.info('Set size: {}'.format(set_size))
                params['set_size'] = set_size
                ds_train, y_train, ds_test, y_test = get_data_by_task(task_name, params)

                train_data = [i[:, 0].reshape(-1, 1) for i in ds_train.records]
                test_data = [i[:, 0].reshape(-1, 1) for i in ds_test.records]
                y_train = y_train
                y_test = y_test
                logging.info('Created dataset')

                train_loader = to_dataloader(permute_set(train_data), y_train, DummyDataset, params['batch_size_train'],
                                             num_workers=params['n_workers'],
                                             shuffle=True,
                                             transform_x=lambda x: x,
                                             collate_fn=collate_func_rnn)
                test_loader = to_dataloader(permute_set(test_data), y_test, DummyDataset, params['batch_size_test'],
                                            num_workers=params['n_workers'],
                                            shuffle=False,
                                            transform_x=lambda x: x,
                                            collate_fn=collate_func_rnn)
                logging.info('Num train {} num test {}'.format(len(train_loader.dataset), len(test_loader.dataset)))

                model = InvarianceModel(theta=nn.Sequential(nn.Linear(1, 25),
                                                            nn.ReLU(inplace=True),
                                                            nn.Linear(25, 50)),
                                        rho=nn.Sequential(nn.Linear(50, 25),
                                                          nn.ReLU(inplace=True),
                                                          nn.Linear(25, 2)),
                                        op=torch.mean)

                model = RNNModel(cell_type=nn.LSTM,
                                 project_in=nn.Linear(1, 64),
                                 project_out=nn.Linear(64, 2),
                                 input_dim=64,
                                 hidden_dim=64,
                                 dropout=0.2,
                                 num_layers=1,
                                 bidirectional=False)

                logging.info('There are {} params'.format(count_parameters(model)))
                logging.info(str(model))
                model = model.to(params['device'])

                criteria = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
                # optimizer = torch.optim.SGD(model.parameters(), momentum=0.9)

                trainer = RegularTrainer(n_epochs=params['n_epochs'],
                                         criteria=criteria,
                                         optimizer=optimizer,
                                         eval_metric=eval.AverageAcc,
                                         device=params['device'],
                                         janossy=params['janossy'],
                                         sire=params['sire'],
                                         sire_alpha=params['sire_alpha'],
                                         early_stop=EarlyStopping(metric_name='met',
                                                                  patience=params['patience'],
                                                                  min_is_better=False), verbose=3)
                trainer.fit(train_loader, test_loader, model)
                test_loss, test_acc, _ = trainer.test(trainer.best_model, test_loader, metric=eval.auc_pytorch)
                results[task_name][set_size].append(test_acc)

            # end task
        # end multiple-runs
    with open(os.path.join(log_dir, '{}_results.json'.format(params['exp_name'])), 'w') as f:
        json.dump(results, f)
