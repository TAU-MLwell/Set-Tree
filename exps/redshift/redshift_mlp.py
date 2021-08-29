from pprint import pformat
from exps.deep_learning import *
from exps.data import get_redmapper_dataset_ii


class RedMapperDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, i):
        return self.x[i, :], self.y[i]

    def __len__(self):
        return len(self.y)


def mean_scatter(y, y_pred):
    return np.average((np.abs(y - y_pred)) / (1 + y))


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, p=0.5):
        super(MLP, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(64, hidden_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, 64),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(64, 1))

    def forward(self, x):
        x = self.net(x)
        return x.flatten()


params = {
    'exp_name': 'mlp_exp_1',
    'seed': 0,
    'normalize': False,

    'input_dim': 17,
    'hidden_dim': 128,
    'output_dim': 2,
    'dropout': 0.5,

    'lr': 1e-3,
    'weight_decay': 0.0,

    'batch_size_train': 128,
    'batch_size_test': 128,
    'device': 'cuda:5',

    'n_epochs': 30,
    'patience': 4,
    'n_workers': 0
}


if __name__ == '__main__':

    log_dir = os.path.join(os.path.abspath('__file__' + '/../'), 'outputs', 'mlp')
    eval.create_logger(log_dir=log_dir,
                       log_name=params['exp_name'],
                       dump=False)

    set_initial_random_seed(params['seed'])
    logging.info(pformat(params))

    # train_data, train_y, test_data, test_y = get_redmapper_dataset(seed=seed, test_size=0.1)
    train_data, train_y, test_data, test_y = get_redmapper_dataset_ii(is_deepsets=True)

    train_loader = DataLoader(RedMapperDataset(train_data, train_y),
                              batch_size=params['batch_size_train'],
                              shuffle=True,
                              num_workers=params['n_workers'])
    test_loader = DataLoader(RedMapperDataset(test_data, test_y),
                             batch_size=params['batch_size_test'],
                             shuffle=False,
                             num_workers=params['n_workers'])

    model = MLP(input_dim=params['input_dim'],
                hidden_dim=params['hidden_dim'],
                p=params['dropout'])

    logging.info('There are {} params'.format(count_parameters(model)))
    logging.info(str(model))
    model = model.to(params['device'])

    criteria = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], betas=(0.9, 0.999), weight_decay=params['weight_decay'])
    #optimizer = optim.SGD(model.parameters(), lr=params['lr'], momentum=0.9)

    trainer = RegularTrainer(n_epochs=params['n_epochs'],
                             criteria=criteria,
                             optimizer=optimizer,
                             eval_metric=eval.AverageReg,
                             device=params['device'],
                             early_stop=EarlyStopping(metric_name='loss',
                                                      patience=params['patience'],
                                                      min_is_better=True), verbose=3)
    trainer.fit(train_loader, test_loader, model)
    test_loss, test_acc = trainer.test(trainer.best_model, test_loader, metric=mean_scatter) # getting (y, preds)
