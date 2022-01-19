import torch
import torch.nn as nn

import random
import numpy as np
import copy
import logging
import exps.eval_utils as eval


class MaskedMSELoss(nn.Module):

    def __init__(self, reduction='mean', ignore_index=-1) -> None:
        super(MaskedMSELoss, self).__init__()

        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        mask = target != self.ignore_index
        return nn.functional.mse_loss(input[mask], target[mask], reduction=self.reduction)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, metric_name, patience=3, min_is_better=True):
        self.metric_name = metric_name
        self.patience = patience
        self.min_is_better = min_is_better
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def reset(self):
        self.counter = 0

    def __call__(self, met_dict):
        score = met_dict[self.metric_name]
        if self.min_is_better:
            score = -score

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class BaseTrainer():
    def __init__(self, n_epochs, criteria, optimizer, eval_metric, device, verbose=3, early_stop=None):
        self.n_epochs = n_epochs
        self.criteria = criteria
        self.optimizer = optimizer
        self.eval_metric = eval_metric
        self.device = device
        self.verbose = verbose
        self.early_stop = early_stop

    def fit(self, train_loader, eval_loader, model):
        self.losses = {'train': [], 'val': []}
        self.mets = {'train': [], 'val': []}

        best_val_loss = 99999
        self.best_epoch = None
        self.best_model = None

        epoch = 0
        for epoch in range(self.n_epochs):
            train_loss, train_met = self.train_single_epoch(model, train_loader)
            val_loss, val_met = self.eval(model, eval_loader)
            if self.verbose > 2:
                logging.info('E[{}] train loss: {:6f} train met: {:6f}'.format(epoch, train_loss, train_met))
                logging.info('E[{}] val loss: {:6f} val met: {:6f}'.format(epoch, val_loss, val_met))

            self.losses['train'].append(train_loss)
            self.losses['val'].append(val_loss)
            self.mets['train'].append(train_met)
            self.mets['val'].append(val_met)

            if best_val_loss >= val_loss:
                self.best_model = copy.deepcopy(model)
                best_val_loss = val_loss
                self.best_epoch = epoch

            if self.early_stop:
                self.early_stop({'loss': val_loss, 'met': val_met})
                if self.early_stop.early_stop:
                    if self.verbose > 0:
                        logging.info('Early stop epoch {} results'.format(self.best_epoch))
                        best_train_loss, best_train_met = self.eval(self.best_model, train_loader)
                        best_val_loss, best_val_met = self.eval(self.best_model, eval_loader)
                        logging.info('E[{}] train loss: {:6f} train met: {:6f}'.format(self.best_epoch, best_train_loss, best_train_met))
                        logging.info('E[{}] val loss: {:6f} val met: {:6f}'.format(self.best_epoch, best_val_loss, best_val_met))
                        break

        # if no early_stop before
        if self.verbose > 0 and epoch == self.n_epochs:
            self.best_epoch = epoch
            self.best_model = model

            logging.info('Finish - best results for epoch {}'.format(epoch))
            logging.info('E[{}] train loss: {:6f} train met: {:6f}'.format(epoch, train_loss, train_met))
            logging.info('E[{}] val loss: {:6f} val met: {:6f}'.format(epoch, val_loss, val_met))

    def train_single_epoch(self, model, train_loader):
        raise NotImplemented

    def eval(self, model, eval_loader):
        raise NotImplemented

    def test(self, model, test_loader, metric=None):
        raise NotImplemented


class DeepSetTrainer(BaseTrainer):
    def __init__(self, n_epochs, criteria, optimizer, eval_metric, device, verbose=3, early_stop=None):
        super().__init__(n_epochs, criteria, optimizer, eval_metric, device, verbose, early_stop)

    def train_single_epoch(self, model, train_loader):
        train_loss = eval.AverageMeter()
        train_metric = self.eval_metric()
        model.train()
        for x, l, y in train_loader:
            self.optimizer.zero_grad()
            x = x.to(self.device)
            y = y.to(self.device)
            out = model(x, l)
            loss = self.criteria(out, y)

            loss.backward()
            self.optimizer.step()
            train_loss.update(loss.item())
            train_metric.update(out.detach().cpu(), y.detach().cpu())
        return train_loss(), train_metric()

    def eval(self, model, eval_loader):
        test_loss = eval.AverageMeter()
        test_metric = self.eval_metric()

        model.eval()
        for x, l, y in eval_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            out = model(x, l)
            loss = self.criteria(out, y)
            test_loss.update(loss.item())
            test_metric.update(out.detach().cpu(), y.detach().cpu())
        return test_loss(), test_metric()

    def test(self, model, test_loader, metric=None):
        test_loss = eval.AverageMeter()
        test_metric = self.eval_metric()

        all_y = []
        all_out = []
        model.eval()
        for x, l, y in test_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            out = model(x, l)
            loss = self.criteria(out, y)
            test_loss.update(loss.item())
            test_metric.update(out.detach().cpu(), y.detach().cpu())

            all_y.append(y.detach().cpu())
            all_out.append(out.detach().cpu())

        all_y = torch.cat(all_y)
        all_out = torch.cat(all_out)
        test_loss = test_loss()
        test_met = test_metric()
        if self.verbose > 2:
            logging.info('test loss: {:6f} test met: {:6f}'.format(test_loss, test_met))
            if metric:
                test_second_met = metric(all_y, all_out)
                logging.info('additional met: {:6f}'.format(test_second_met))
                return test_loss, test_met, test_second_met
        return test_loss, test_met


class RegularTrainer(BaseTrainer):
    def __init__(self, n_epochs, criteria, optimizer, eval_metric, device, verbose=3, early_stop=None,
                 janossy=False, sire=False, sire_alpha=0.1):
        super().__init__(n_epochs, criteria, optimizer, eval_metric, device, verbose, early_stop)
        self.janossy = janossy
        self.sire = sire
        self.sire_alpha = sire_alpha

    def train_single_epoch(self, model, train_loader):
        train_loss = eval.AverageMeter()
        train_metric = self.eval_metric()

        model.train()
        for x, y in train_loader:
            self.optimizer.zero_grad()

            if self.janossy:
                x = torch.stack([t[torch.randperm(len(t)), :] for t in x])

            x = x.to(self.device)
            y = y.to(self.device)
            out = model(x)
            loss = self.criteria(out, y)

            if self.sire:
                loss += self.sire_alpha * self.sire_regularize(x, model)

            loss.backward()
            self.optimizer.step()
            train_loss.update(loss.item())
            train_metric.update(out.detach().cpu(), y.detach().cpu())
        return train_loss(), train_metric()

    def eval(self, model, eval_loader):
        test_loss = eval.AverageMeter()
        test_metric = self.eval_metric()

        model.eval()
        with torch.no_grad():
            for x, y in eval_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                out = model(x)
                loss = self.criteria(out, y)
                test_loss.update(loss.item())
                test_metric.update(out.detach().cpu(), y.detach().cpu())
        return test_loss(), test_metric()

    def test_janossy(self, model, test_loader, k, metric=None):
        test_metric = eval.AverageMeter()

        all_y = []
        all_out = []
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                outs = []
                for _ in range(k):
                    x_perm = torch.stack([t[torch.randperm(len(t)), :] for t in x])
                    x_perm = x_perm.to(self.device)
                    out = model(x_perm)
                    outs.append(out.detach().cpu().argmax(1))
                outs = torch.stack(outs, 1)
                preds = []
                for o in outs:
                    unique, counts = torch.unique(o, return_counts=True)
                    preds.append(unique[counts.argmax()].item())
                test_metric.update((torch.tensor(preds).long() == y).float().mean())

                all_y.append(y.detach().cpu())
                all_out.append(torch.tensor(preds).long())

        all_y = torch.cat(all_y)
        all_out = torch.cat(all_out)
        test_met = test_metric().item()
        if self.verbose > 2:
            logging.info('test loss: {:6f} test met: {:6f}'.format(0.0, test_met))
            if metric:
                test_second_met = metric(all_y, all_out)
                logging.info('additional met: {:6f}'.format(test_second_met))
                return 0.0, test_met, test_second_met
        return 0.0, test_met

    def test(self, model, test_loader, metric=None):
        test_loss = eval.AverageMeter()
        test_metric = self.eval_metric()

        all_y = []
        all_out = []
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                out = model(x)
                loss = self.criteria(out, y)
                test_loss.update(loss.item())
                test_metric.update(out.detach().cpu(), y.detach().cpu())

                all_y.append(y.detach().cpu())
                all_out.append(out.detach().cpu())

        all_y = torch.cat(all_y)
        all_out = torch.cat(all_out)
        test_loss = test_loss()
        test_met = test_metric()
        if self.verbose > 2:
            logging.info('test loss: {:6f} test met: {:6f}'.format(test_loss, test_met))
            if metric:
                test_second_met = metric(all_y, all_out)
                logging.info('additional met: {:6f}'.format(test_second_met))
                return test_loss, test_met, test_second_met
        return test_loss, test_met

    def sire_regularize(self, x, model, inv_chunk_size=1):
        shape_0, shape_1, _ = x.shape
        device = x.device

        idx = torch.randint(low=0, high=shape_1 - 2*inv_chunk_size, size=()).to(device)

        prefix = torch.narrow(x, dim=1, start=0, length=idx)
        suffix_a = torch.narrow(x, dim=1, start=idx, length=2 * inv_chunk_size)

        reverse_list = np.arange(inv_chunk_size, 2*inv_chunk_size).tolist() + np.arange(inv_chunk_size).tolist()
        suffix_b = suffix_a[:, reverse_list, :]

        seq_a = torch.cat((prefix, suffix_a), dim=1)
        seq_b = torch.cat((prefix, suffix_b), dim=1)

        output_a, hidden_a = model.get_rnn_output(seq_a)
        output_b, hidden_b = model.get_rnn_output(seq_b)

        hidden_a = hidden_a[0]
        hidden_b = hidden_b[0]

        return (hidden_a - hidden_b).pow(2).sum() / shape_0


def set_initial_random_seed(random_seed):
    if random_seed != -1:
        np.random.seed(random_seed)
        torch.random.manual_seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

