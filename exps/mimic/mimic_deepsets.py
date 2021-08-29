from exps.deep_learning import *
from pprint import pformat
from exps.data import get_mimic_data_fold
import exps.eval_utils as eval

if __name__ == '__main__':
    params = {'exp_name': 'strict_num_dss_sum',
              'seed': 0,
              'type': 'strict',
              'num': 1,
              'normalize': True,
              'input_dim': 18,
              'output_dim': 2,
              'dropout': 0.2,
              'batch_size_train': 64,
              'batch_size_test': 128,
              'lr': 1e-3,
              'weight_decay': 0.0,

              'device': 'cuda:0',
              'n_epochs': 100,
              'patience': 5,
              'n_workers': 4,
              'n_seeds': 5}

    log_dir = os.path.join(os.path.abspath('__file__' + '/../'), 'outputs', 'DSS')
    eval.create_logger(log_dir=log_dir,
                       log_name=params['exp_name'],
                       dump=True)

    mets = []
    for seed in range(params['n_seeds']):
        set_initial_random_seed(seed)
        logging.info('Seed {}'.format(seed))
        logging.info(pformat(params))

        dataset = get_mimic_data_fold(params['type'], params['num'])
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
                                     collate_fn=collate_func)
        test_loader = to_dataloader(permute_set(test_data), y_test, DummyDataset, params['batch_size_test'],
                                    num_workers=params['n_workers'],
                                    shuffle=False,
                                    transform_x=normalize(m, s) if params['normalize'] else lambda x: x,
                                    collate_fn=collate_func)

        logging.info('Finish preprocessing')
        if params['normalize']:
            logging.info('Normalize records:')
            logging.info('mean: {}'.format(m))
            logging.info('std: {}'.format(m))

        logging.info('Num train {} num test {}'.format(len(train_loader.dataset), len(test_loader.dataset)))

        # model = InvarianceModelVaryingSizes(theta=nn.Sequential(nn.Linear(params['input_dim'], 50),
        #                                             nn.ReLU(inplace=True),
        #                                             nn.Linear(50, 100),
        #                                             nn.ReLU(inplace=True),
        #                                             nn.Linear(100, 50)),
        #                                     rho=nn.Sequential(nn.Linear(50, 30),
        #                                                       nn.ReLU(inplace=True),
        #                                                       nn.Linear(30, 30),
        #                                                       nn.ReLU(inplace=True),
        #                                                       nn.Linear(30, 10),
        #                                                       nn.ReLU(inplace=True),
        #                                                       nn.Linear(10, 2)),
        #                                     op=torch.mean)

        model = DSSInvarianceModelVaryingSizes([params['input_dim'], 50, 100, 50],
                                   rho=nn.Sequential(nn.Linear(50, 30),
                                                              nn.ReLU(inplace=True),
                                                              nn.Linear(30, 30),
                                                              nn.ReLU(inplace=True),
                                                              nn.Linear(30, 10),
                                                              nn.ReLU(inplace=True),
                                                              nn.Linear(10, 2)),
                                   drop_rate=None,
                                   op=torch.sum)


        logging.info('There are {} params'.format(count_parameters(model)))
        logging.info(str(model))
        model = model.to(params['device'])

        criteria = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        #optimizer = torch.optim.SGD(model.parameters(), momentum=0.9)

        trainer = DeepSetTrainer(n_epochs=params['n_epochs'],
                                 criteria=criteria,
                                 optimizer=optimizer,
                                 eval_metric=eval.AverageAcc,
                                 device=params['device'],
                                 early_stop=EarlyStopping(metric_name='met',
                                                      patience=params['patience'],
                                                      min_is_better=False), verbose=3)
        trainer.fit(train_loader, test_loader, model)
        test_loss, test_acc, test_auc = trainer.test(trainer.best_model, test_loader, metric=eval.auc_pytorch)
        mets.append({'test_loss': test_loss, 'test_acc': test_acc, 'test_auc': test_auc})

    eval.save_json({'mets': mets}, os.path.join(log_dir, '{}_multi_results.json'.format(params['exp_name'])))

    loss = np.array([item['test_loss'] for item in mets])
    logging.info('Loss: mean: {:.4f} | std: {:.4f}'.format(loss.mean(), loss.std()))
    acc = np.array([item['test_acc'] for item in mets])
    logging.info('Acc: mean: {:.4f} | std: {:.4f}'.format(acc.mean(), acc.std()))
    acc = np.array([item['test_auc'] for item in mets])
    logging.info('AUC: mean: {:.4f} | std: {:.4f}'.format(acc.mean(), acc.std()))

