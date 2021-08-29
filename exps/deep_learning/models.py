import torch
import torch.autograd as autograd
import torch.nn as nn

def protected_max(x, dim):
    return torch.max(x, dim=dim)[0]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ModelWarper(nn.Module):
    def __init__(self, layers):
        super(ModelWarper, self).__init__()
        self.layers = layers

    def forward(self, x):
        out = self.layers(x)
        return out


class DSSLinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, op=torch.sum):
        super(DSSLinearLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.shared_fc = nn.Linear(input_dim, output_dim)
        self.shared_bn = nn.BatchNorm1d(output_dim)
        self.op = op

    def forward(self, x):
        x_i = self.fc(x)
        bs, n, d = x_i.shape
        x_i = self.bn(x_i.view(bs * n, d)).view(bs, n, d)

        x_s = self.shared_fc(self.op(x, 1))
        x_s = self.shared_bn(x_s)
        o = x_s.unsqueeze(1).repeat([1, n, 1]) + x_i
        return o


class DSSLinearLayerVaringSizes(nn.Module):
    def __init__(self, input_dim, output_dim, op=torch.sum):
        super(DSSLinearLayerVaringSizes, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.shared_fc = nn.Linear(input_dim, output_dim)
        self.shared_bn = nn.BatchNorm1d(output_dim)

        self.op = op

    def forward(self, x, l):
        x_i = self.fc(x)
        if x.ndim == 3:
            bs, n, d = x_i.shape
            x_i = self.bn(x_i.view(bs * n, d)).view(bs, n, d)
        else:
            x_i = self.bn(x_i)
        splits = torch.split(x, split_size_or_sections=l, dim=0)
        x_m = torch.stack([self.op(i, 0) for i in splits])
        x_s = self.shared_fc(x_m)
        x_s = self.shared_bn(x_s)
        o = x_s.unsqueeze(1).repeat_interleave(torch.tensor(list(l)).to(x.device), 0).squeeze(1) + x_i
        return o


class DSSInvarianceModel(nn.Module):
    def __init__(self, hidden_states, rho, drop_rate=None, op=torch.sum):
        super(DSSInvarianceModel, self).__init__()

        self.op = op
        layers = []
        for i in range(len(hidden_states) - 1):
            layers.append(DSSLinearLayer(hidden_states[i], hidden_states[i + 1], op))
            layers.append(nn.ReLU(inplace=True))
            if drop_rate != None:
                layers.append(nn.Dropout(drop_rate))

        self.layers = nn.Sequential(*layers[:-2 if drop_rate != None else -1])
        self.rho = rho

    def forward(self, x):
        x = self.layers(x)
        xs = torch.sum(x, 1) # the authors claim to always use sum
        out = self.rho(xs)
        return out


class DSSInvarianceModelVaryingSizes(nn.Module):
    def __init__(self, hidden_states, rho, drop_rate=None, op=torch.sum):
        super(DSSInvarianceModelVaryingSizes, self).__init__()

        self.op = op

        self.dds_1 = DSSLinearLayerVaringSizes(hidden_states[0], hidden_states[1], op)
        self.dds_2 = DSSLinearLayerVaringSizes(hidden_states[1], hidden_states[2], op)
        self.dds_3 = DSSLinearLayerVaringSizes(hidden_states[2], hidden_states[3], op)
        self.rho = rho

    def forward(self, x, l):
        x = self.dds_1(x, l)
        x = self.dds_2(x, l)
        x = self.dds_3(x, l)

        splits = torch.split(x, split_size_or_sections=l, dim=0)
        x_m = torch.stack([torch.sum(i, 0) for i in splits]) # the authors claim to always use sum
        out = self.rho(x_m)
        return out


class InvarianceModel(nn.Module):
    def __init__(self, theta, rho, op=torch.sum):
        super(InvarianceModel, self).__init__()
        self.op = op
        self.theta = theta
        self.rho = rho

    def forward(self, x):
        x = self.theta(x)
        xs = self.op(x, 1)
        out = self.rho(xs)
        return out


class InvarianceModelVaryingSizes(nn.Module):
    def __init__(self, theta, rho, op=torch.sum):
        super(InvarianceModelVaryingSizes, self).__init__()
        self.op = op
        self.theta = theta
        self.rho = rho

    def forward(self, x, l):
        x = self.theta(x)
        splits = torch.split(x, split_size_or_sections=l, dim=0)
        xm = torch.stack([self.op(s, 0) for s in splits])
        out = self.rho(xm)
        return out


class PermEqui1_mean(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(PermEqui1_mean, self).__init__()
    self.Gamma = nn.Linear(in_dim, out_dim)

  def forward(self, x):
    xm = x.mean(1, keepdim=True)
    x = self.Gamma(x-xm)
    return x


class EfficientEquivariantLayer(nn.Module):
  def __init__(self, input_dim, output_dim, op):
    super(EfficientEquivariantLayer, self).__init__()

    self.Gamma = nn.Linear(input_dim, output_dim)
    self.op = op

  def forward(self, x, l):
      splits = torch.split(x, split_size_or_sections=l, dim=0)
      xm = torch.stack([self.op(s, 0) for s in splits])
      xm = torch.repeat_interleave(xm, torch.tensor(l).to(x.device), dim=0)

      out = self.Gamma(x - xm)
      return out


class EquivariantLayer(nn.Module):
  def __init__(self, input_dim, output_dim, op):
    super(EquivariantLayer, self).__init__()

    self.Gamma = nn.Linear(input_dim, output_dim)
    self.Lambda = nn.Linear(input_dim, output_dim, bias=False)
    self.op = op

  def forward(self, x, l):
      splits = torch.split(x, split_size_or_sections=l, dim=0)
      xs = torch.stack([self.op(s, 0) for s in splits])
      xm = self.Lambda(xs)

      xm = torch.repeat_interleave(xm, torch.tensor(l).to(x.device), dim=0)
      x = self.Gamma(x)
      out = x - xm
      return out


def create_rnn_model(cell_type, project_in_dict, project_out_dict, rnn_input_dim, rnn_hidden_dim, dropout=0.2, num_layers=1, bidirectional=False):

    if isinstance(project_in_dict['hidden_dim'], int):
        project_in = nn.Linear(project_in_dict['input_dim'], project_in_dict['hidden_dim'])
    else:
        hidden_dims = project_in_dict['hidden_dim']
        project_in = []
        project_in.append(nn.Linear(project_in_dict['input_dim'], hidden_dims[0]))
        project_in.append(nn.ReLU(inplace=True))

        for k in range(len(hidden_dims) - 1):
            project_in.append(nn.Linear(hidden_dims[k], hidden_dims[k + 1]))
            project_in.append(nn.ReLU(inplace=True))
        project_in.pop()
        project_in = nn.Sequential(*project_in)

    if isinstance(project_out_dict['hidden_dim'], int):
        project_out = nn.Linear(project_out_dict['hidden_dim'], project_out_dict['output_dim'])
    else:
        hidden_dims = project_out_dict['hidden_dim']
        project_out = []
        for k in range(len(hidden_dims) - 1):
            project_out.append(nn.Linear(hidden_dims[k], hidden_dims[k + 1]))
            project_out.append(nn.ReLU(inplace=True))
        project_out.append(nn.Linear(hidden_dims[-1], project_out_dict['output_dim'], ))
        project_out = nn.Sequential(*project_out)
    return RNNModel(cell_type, project_in, project_out, rnn_input_dim, rnn_hidden_dim, dropout, num_layers, bidirectional)


class RNNModel(nn.Module):

    def __init__(self, cell_type, project_in, project_out, input_dim, hidden_dim, dropout=0.2, num_layers=1, bidirectional=False):
        super(RNNModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_directions = 2 if bidirectional else 1

        self.project_in = project_in
        self.rnn = cell_type(input_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional)

        if self.rnn._get_name() == 'GRU':
            if bidirectional:
                self.fnc = self._process_gru_bi
            else:
                self.fnc = self._process_gru
        else:
            if bidirectional:
                self.fnc = self._process_lstm_bi
            else:
                self.fnc = self._process_lstm

        self.dropout = nn.Dropout(p=dropout)
        self.project_out = project_out

    def init_hidden(self, batch_size, device):
        n = self.n_directions * self.num_layers
        if self.rnn._get_name() == 'GRU':
            return autograd.Variable(torch.randn(n, batch_size, self.hidden_dim)).to(device)

        else:
            return (autograd.Variable(torch.randn(n, batch_size, self.hidden_dim)).to(device),
                    autograd.Variable(torch.randn(n, batch_size, self.hidden_dim)).to(device))

    def _process_lstm_bi(self, ht, batch_size):
        return torch.cat((ht[0][0], ht[0][1]), dim=1)

    def _process_lstm(self, ht, batch_size):
        return ht[0].view(batch_size, self.n_directions * self.hidden_dim)

    def _process_gru(self, ht, batch_size):
        return ht.view(batch_size, self.n_directions * self.hidden_dim)

    def _process_gru_bi(self, ht, batch_size):
        return torch.cat((ht[0, :, :], ht[1, :, :]), dim=1)

    def forward(self, batch):
        batch = self.project_in(batch)
        batch_size = batch.shape[0]

        batch = batch.permute(1, 0, 2) # (seq_len, batch, input_size)
        self.hidden = self.init_hidden(batch_size, batch.device) # num_layers * num_directions, batch, hidden_size
        outputs, ht = self.rnn(batch, self.hidden)
        output = self.dropout(self.fnc(ht, batch_size))
        output = self.project_out(output)
        return output

    def get_rnn_output(self, batch):
        batch = self.project_in(batch)
        batch_size = batch.shape[0]

        batch = batch.permute(1, 0, 2) # (seq_len, batch, input_size)
        self.hidden = self.init_hidden(batch_size, batch.device) # num_layers * num_directions, batch, hidden_size
        outputs, ht = self.rnn(batch, self.hidden)
        return outputs, ht
