from pprint import pformat
from exps.deep_learning import *
import exps.eval_utils as eval
from exps.data import get_mimic_data_fold


if __name__ == '__main__':
    params = {'exp_name': 'lstm_sire_multiple_mild',
              'seed': 0,
              'type': 'mild',
              'num': 1,

              'normalize': True,
              'janossy': False,
              'sire': True,
              'sire_alpha': 0.1,

              'input_dim': 18,
              'hidden_dim': [30, 50],
              'output_dim': 2,
              'dropout': 0.2,
              'num_layers': 1,
              'bidirectional': False,

              'batch_size_train': 64,
              'batch_size_test': 128,
              'device': 'cuda:3',
              'n_epochs': 30,
              'patience': 4,
              'n_workers': 4,
              'n_seeds': 1,
              }

    log_dir = os.path.join(os.path.abspath('__file__' + '/../'), 'outputs', 'SIRE')
    eval.create_logger(log_dir=log_dir,
                       log_name=params['exp_name'],
                       dump=False)

    mets = []
    for seed in range(params['n_seeds']):
        set_initial_random_seed(seed)
        logging.info('Seed {}'.format(seed))
        logging.info(pformat(params))

        dataset = get_mimic_data_fold(params['type'], seed)
        train_data = dataset['ds_train'].records
        test_data = dataset['ds_test'].records
        y_train = dataset['y_train']
        y_test = dataset['y_test']
        logging.info('Created dataset')

        m, s = get_stats(train_data)
        train_loader = to_dataloader(permute_set(train_data), y_train, DummyDataset, params['batch_size_train'],
                                     num_workers=params['n_workers'],
                                     shuffle=True,
                                     transform_x=normalize(m, s) if params['normalize'] else lambda x: x,
                                     collate_fn=collate_func_rnn)
        test_loader = to_dataloader(permute_set(test_data), y_test, DummyDataset, params['batch_size_test'],
                                    num_workers=params['n_workers'],
                                    shuffle=False,
                                    transform_x=normalize(m, s) if params['normalize'] else lambda x: x,
                                    collate_fn=collate_func_rnn)

        logging.info('Finish preprocessing')
        if params['normalize']:
            logging.info('Normalize records:')
            logging.info('mean: {}'.format(m))
            logging.info('std: {}'.format(m))

        logging.info('Num train {} num test {}'.format(len(train_loader.dataset), len(test_loader.dataset)))

        model = create_rnn_model(cell_type=nn.LSTM,
                                 project_in_dict={'input_dim': 18,
                                                  'hidden_dim': [30, 50]},
                                 project_out_dict={'output_dim': 2,
                                                   'hidden_dim': 50},
                                 rnn_input_dim=50,
                                 rnn_hidden_dim=50,
                                 dropout=params['dropout'],
                                 num_layers=params['num_layers'],
                                 bidirectional=params['bidirectional'])

        logging.info('There are {} params'.format(count_parameters(model)))
        logging.info(str(model))
        model = model.to(params['device'])

        criteria = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        #optimizer = torch.optim.SGD(model.parameters(), momentum=0.9)

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
        if params['janossy']:
            test_loss, test_acc = trainer.test_janossy(trainer.best_model, test_loader, k=20)
            mets.append({'test_loss': test_loss, 'test_acc': test_acc, 'test_auc': 0.0})
        else:
            test_loss, test_acc, test_auc = trainer.test(trainer.best_model, test_loader, metric=eval.auc_pytorch)
            mets.append({'test_loss': test_loss, 'test_acc': test_acc, 'test_auc': test_auc})

    eval.save_json({'mets': mets}, os.path.join(log_dir, '{}_multi_results.json'.format(params['exp_name'])))

    loss = np.array([item['test_loss'] for item in mets])
    logging.info('Loss: mean: {:.4f} | std: {:.4f}'.format(loss.mean(), loss.std()))
    acc = np.array([item['test_acc'] for item in mets])
    logging.info('Acc: mean: {:.4f} | std: {:.4f}'.format(acc.mean(), acc.std()))
    acc = np.array([item['test_auc'] for item in mets])
    logging.info('AUC: mean: {:.4f} | std: {:.4f}'.format(acc.mean(), acc.std()))

    # torch.save({'params': params, 'state_dict': model.state_dict(), 'time': str(get_datetime())},
    #            os.path.join(log_dir, exp_name + '_model.pkl'))
