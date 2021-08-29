from pprint import pformat
from exps.deep_learning import *
from exps.data import get_redmapper_dataset_ii


class RedMapperDataset(Dataset):
    def __init__(self, x, y):
        self.x = [torch.FloatTensor(i) for i in x]
        self.y = [torch.FloatTensor(i) for i in y]

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.y)


def masked_mean_scatter(y, y_pred, ignore_index=-1):
    mask = y != ignore_index
    y = y[mask]
    y_pred = y_pred[mask]
    return mean_scatter(y, y_pred)


def mean_scatter(y, y_pred):
    return np.average((np.abs(y - y_pred)) / (1 + y))


params = {
    'exp_name': 'sire_2',

    'seed': 1,
    'normalize': False,

    'janossy': False,
    'sire': True,
    'sire_alpha': 0.1,

    'input_dim': 18,
    'output_dim': 1,
    'dropout': 0.5,

    'batch_size_train': 128,
    'batch_size_test': 128,

    'lr': 1e-3,
    'weight_decay': 0.0,

    'device': 'cuda:3',
    'n_epochs': 100,
    'patience': 20,
    'n_workers': 4
}

if __name__ == '__main__':

    log_dir = os.path.join(os.path.abspath('__file__' + '/../'), 'outputs', 'SIRE')
    eval.create_logger(log_dir=log_dir,
                       log_name=params['exp_name'],
                       dump=True)

    logging.info(pformat(params))
    set_initial_random_seed(params['seed'])

    train_data, train_y, test_data, test_y = get_redmapper_dataset_ii(is_deepsets=False)
    m, s = get_stats(train_data)

    train_loader = to_dataloader(permute_set(train_data), train_y, DummyRegressionDataset, params['batch_size_train'],
                                 num_workers=params['n_workers'],
                                 shuffle=True,
                                 transform_x=normalize(m, s) if params['normalize'] else lambda x: x,
                                 collate_fn=collate_func_rnn_regression)
    test_loader = to_dataloader(permute_set(test_data), test_y, DummyRegressionDataset, params['batch_size_test'],
                                num_workers=params['n_workers'],
                                shuffle=False,
                                transform_x=normalize(m, s) if params['normalize'] else lambda x: x,
                                collate_fn=collate_func_rnn_regression)

    logging.info('Finish preprocessing')
    if params['normalize']:
        logging.info('Normalize records:')
        logging.info('mean: {}'.format(m))
        logging.info('std: {}'.format(m))
    logging.info('Num train {} num test {}'.format(len(train_loader.dataset), len(test_loader.dataset)))

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

    criteria = MaskedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    trainer = RegularTrainer(n_epochs=params['n_epochs'],
                             criteria=criteria,
                             optimizer=optimizer,
                             eval_metric=eval.MaskedAverageReg,
                             device=params['device'],
                             janossy=params['janossy'],
                             sire=params['sire'],
                             sire_alpha=params['sire_alpha'],
                             early_stop=EarlyStopping(metric_name='loss',
                                                  patience=params['patience'],
                                                  min_is_better=True), verbose=3)
    trainer.fit(train_loader, test_loader, model)
    test_loss, test_acc = trainer.test(trainer.best_model, test_loader, metric=masked_mean_scatter)
