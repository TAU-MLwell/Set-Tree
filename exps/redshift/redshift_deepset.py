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


class mymax():

    def __call__(self, x, axis):
        return torch.max(x, axis=axis)[0]

    def __name__(self):
        return 'max'


class EquivariantModelVaryingSizes(nn.Module):
    def __init__(self, block, input_dim, output_dim, op):
        super(EquivariantModelVaryingSizes, self).__init__()
        logging.info('Operator is {}'.format(op.__name__))

        self.eq_1 = block(input_dim=input_dim, output_dim=128, op=op)
        self.eq_2 = block(input_dim=128, output_dim=128, op=op)
        self.eq_3 = block(input_dim=128, output_dim=128, op=op)
        self.eq_4 = block(input_dim=128, output_dim=output_dim, op=op)

        self.do_1 = nn.Dropout(0.5)
        #self.do_2 = nn.Dropout(0.5)
        #self.do_3 = nn.Dropout(0.5)

    def forward(self, x, l):
        x = self.eq_1(x, l)
        x = torch.tanh(x)
        x = self.do_1(x)

        x = self.eq_2(x, l)
        x = torch.tanh(x)
        x = self.do_1(x)

        x = self.eq_3(x, l)
        x = torch.tanh(x)
        x = self.do_1(x)

        out = self.eq_4(x, l)
        return out.flatten()


class DSSEquivariantModelVaryingSizes(nn.Module):
    def __init__(self, block, input_dim, output_dim, op):
        super(DSSEquivariantModelVaryingSizes, self).__init__()
        logging.info('Operator is {}'.format(op.__name__))

        self.eq_1 = block(input_dim=input_dim, output_dim=128, op=op)
        self.eq_2 = block(input_dim=128, output_dim=128, op=op)
        self.eq_3 = block(input_dim=128, output_dim=128, op=op)
        self.eq_4 = block(input_dim=128, output_dim=output_dim, op=op)

        self.do_1 = nn.Dropout(0.5)
        #self.do_2 = nn.Dropout(0.5)
        #self.do_3 = nn.Dropout(0.5)

    def forward(self, x, l):
        x = self.eq_1(x, l)
        x = torch.tanh(x)
        x = self.do_1(x)

        x = self.eq_2(x, l)
        x = torch.tanh(x)
        x = self.do_1(x)

        x = self.eq_3(x, l)
        x = torch.tanh(x)
        x = self.do_1(x)

        out = self.eq_4(x, l)
        return out.flatten()


params = {
    'exp_name': 'dss_sum_2',

    'seed': 1,
    'normalize': False,

    'input_dim': 17,
    'output_dim': 1,
    'dropout': 0.5,

    'batch_size_train': 128,
    'batch_size_test': 128,

    'lr': 1e-3,
    'weight_decay': 0.0,

    'device': 'cuda:4',
    'n_epochs': 100,
    'patience': 10,
    'n_workers': 4
}

if __name__ == '__main__':

    log_dir = os.path.join(os.path.abspath('__file__' + '/../'), 'outputs', 'DSS')
    eval.create_logger(log_dir=log_dir,
                       log_name=params['exp_name'],
                       dump=True)

    logging.info(pformat(params))
    set_initial_random_seed(params['seed'])

    train_data, train_y, test_data, test_y = get_redmapper_dataset_ii(is_deepsets=True)
    m, s = get_stats(train_data)

    train_loader = to_dataloader(permute_set(train_data), train_y, EquivariantDummyDataset, params['batch_size_train'],
                                 num_workers=params['n_workers'],
                                 shuffle=True,
                                 transform_x=normalize(m, s) if params['normalize'] else lambda x: x,
                                 collate_fn=collate_func_equivariant)
    test_loader = to_dataloader(permute_set(test_data), test_y, EquivariantDummyDataset, params['batch_size_test'],
                                num_workers=params['n_workers'],
                                shuffle=False,
                                transform_x=normalize(m, s) if params['normalize'] else lambda x: x,
                                collate_fn=collate_func_equivariant)

    logging.info('Finish preprocessing')
    if params['normalize']:
        logging.info('Normalize records:')
        logging.info('mean: {}'.format(m))
        logging.info('std: {}'.format(m))
    logging.info('Num train {} num test {}'.format(len(train_loader.dataset), len(test_loader.dataset)))

    # model = EquivariantModelVaryingSizes(block=EquivariantLayer, # EfficientEquivariantLayer
    #                                      input_dim=params['input_dim'],
    #                                      output_dim=params['output_dim'],
    #                                      op=torch.sum)

    model = DSSEquivariantModelVaryingSizes(block=DSSLinearLayerVaringSizes,
                                            input_dim=params['input_dim'],
                                            output_dim=params['output_dim'],
                                            op=torch.sum)

    logging.info('There are {} params'.format(count_parameters(model)))
    logging.info(str(model))
    model = model.to(params['device'])

    criteria = MaskedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    trainer = DeepSetTrainer(n_epochs=params['n_epochs'],
                             criteria=criteria,
                             optimizer=optimizer,
                             eval_metric=eval.MaskedAverageReg,
                             device=params['device'],
                             early_stop=EarlyStopping(metric_name='loss',
                                                  patience=params['patience'],
                                                  min_is_better=True), verbose=3)
    trainer.fit(train_loader, test_loader, model)
    test_mets = trainer.test(trainer.best_model, test_loader, metric=masked_mean_scatter)
