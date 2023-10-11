from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

from VGDF.model.base import Module, Swish


class Normalizer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim        = dim
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('std', torch.zeros(dim))

    def fit(self, X: np.array) -> None:
        assert len(X.shape)      == 2
        assert X.shape[1]   == self.dim
        device  =   self.mean.device
        self.mean.data.copy_(
            torch.from_numpy(np.mean(X, axis=0, keepdims=False)).float().to(device)
        )
        self.std.data.copy_(
            torch.from_numpy(np.std(X, axis=0, keepdims=False)).float().to(device)
        )
        self.std[self.std < 1e-12]  =   1.0

    def transform(self, x: Union[np.array, torch.tensor]) -> Union[np.array, torch.tensor]:
        if isinstance(x, np.ndarray):
            device  =   self.mean.device
            x       =   torch.from_numpy(x).float().to(device)
            return ((x - self.mean) / self.std).cpu().numpy()
        elif isinstance(x, torch.Tensor):
            return ((x - self.mean) / self.std)



def init_weights(m):
    def truncated_normal_init(
        t:  nn.Module, 
        mean:   float = 0.0, 
        std:    float = 0.01
    ):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear or isinstance(m, EnsembleFC):
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)


class EnsembleFC(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, weight_decay: float = 0., bias: bool = True) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class EnsembleModel(nn.Module):
    def __init__(
        self, 
        state_size:     int, 
        action_size:    int, 
        reward_size:    int, 
        ensemble_size:  int, 
        hidden_size:    int   = 200, 
        learning_rate:  float = 1e-3, 
        use_decay:      bool  = False,
        device:         str   = 'cuda'
    ):
        super(EnsembleModel, self).__init__()
        self.device      = device
        self.hidden_size = hidden_size
        self.output_dim  = state_size + reward_size
        # trunk layers
        self.nn1 = EnsembleFC(state_size + action_size, hidden_size, ensemble_size, weight_decay=0.000025)
        self.nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00005)
        self.nn3 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.nn4 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.use_decay = use_decay
        # Add variance output
        self.nn5 = EnsembleFC(hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001)
        # min / max log var bounds
        self.max_logvar = nn.Parameter((torch.ones((1, self.output_dim)).float() / 2).to(device), requires_grad=False)
        self.min_logvar = nn.Parameter((-torch.ones((1, self.output_dim)).float() * 10).to(device), requires_grad=False)
        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # weight init
        self.apply(init_weights)
        self.swish = Swish()

    def forward(
        self, 
        x:           torch.tensor, 
        ret_log_var: bool = False
    ):
        nn1_output = self.swish(self.nn1(x))
        nn2_output = self.swish(self.nn2(nn1_output))
        nn3_output = self.swish(self.nn3(nn2_output))
        nn4_output = self.swish(self.nn4(nn3_output))
        nn5_output = self.nn5(nn4_output)

        mean    = nn5_output[:, :, :self.output_dim]
        logvar  = nn5_output[:, :, self.output_dim:]

        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        if ret_log_var:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
        return decay_loss

    def loss(
        self, 
        mean:           torch.tensor, 
        logvar:         torch.tensor, 
        labels:         torch.tensor, 
        inc_var_loss:   bool = True
    ):
        """
            mean, logvar: [ensemble_size,  batch_size, |S| + |A|]
            labels:       [ensemble_size,  batch_size, |S| + 1]
        """
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3
        inv_var = torch.exp(-logvar)
        # Average over batch and dim, sum over ensembles.
        if inc_var_loss:
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def train(self, loss):
        self.optimizer.zero_grad()
        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        if self.use_decay:
            loss += self.get_decay_loss()
        loss.backward()
        self.optimizer.step()


class EnsembleDynamicsModel(Module):
    def __init__(
        self, 
        network_size:   int, 
        elite_size:     int, 
        state_size:     int, 
        action_size:    int, 
        reward_size:    int = 1, 
        hidden_size:    int = 200, 
        use_decay:      bool= False,
        device:         str = 'cuda',
    ):
        super(EnsembleDynamicsModel, self).__init__()
        self.network_size = network_size
        self.elite_size = elite_size
        self.model_list = []
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.network_size = network_size
        self.elite_model_idxes = []
        self.ensemble_model = EnsembleModel(state_size, action_size, reward_size, network_size, hidden_size, use_decay=use_decay, device=device)
        self.scaler = Normalizer(dim=state_size + action_size)
        self.device = device

    def train(
        self, 
        inputs:                     np.array, 
        labels:                     np.array, 
        batch_size:                 int     = 256, 
        holdout_ratio:              float   = 0., 
        max_epochs_since_update:    int     = 5
    ):
        self._max_epochs_since_update   = max_epochs_since_update
        self._epochs_since_update       = 0
        self._state                     = {}
        self._snapshots                 = {i: (None, 1e10) for i in range(self.network_size)}

        num_holdout     = int(inputs.shape[0] * holdout_ratio)
        permutation     = np.random.permutation(inputs.shape[0])
        inputs, labels  = inputs[permutation], labels[permutation]

        train_inputs, train_labels      = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels  = inputs[:num_holdout], labels[:num_holdout]

        self.scaler.fit(train_inputs)
        train_inputs    = self.scaler.transform(train_inputs)
        holdout_inputs  = self.scaler.transform(holdout_inputs)

        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(self.device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(self.device)
        holdout_inputs = holdout_inputs[None, :, :].repeat([self.network_size, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat([self.network_size, 1, 1])
        # for log
        all_holdout_losses  =   []

        for epoch in itertools.count():
            train_idx = np.vstack([np.random.permutation(train_inputs.shape[0]) for _ in range(self.network_size)])
            for start_pos in range(0, train_inputs.shape[0], batch_size):
                idx = train_idx[:, start_pos: min(start_pos + batch_size, train_inputs.shape[0])]
                train_input = torch.from_numpy(train_inputs[idx]).float().to(self.device)
                train_label = torch.from_numpy(train_labels[idx]).float().to(self.device)
                losses = []
                mean, logvar = self.ensemble_model(train_input, ret_log_var=True)
                loss, _ = self.ensemble_model.loss(mean, logvar, train_label)
                self.ensemble_model.train(loss)
                losses.append(loss)

            with torch.no_grad():
                holdout_mean, holdout_logvar    = self.ensemble_model(holdout_inputs, ret_log_var = True)
                _, holdout_mse_losses           = self.ensemble_model.loss(holdout_mean, holdout_logvar, holdout_labels, inc_var_loss=False)
                holdout_mse_losses              = holdout_mse_losses.detach().cpu().numpy()
                all_holdout_losses.append(np.mean(holdout_mse_losses))
                # rank and select the elite models
                sorted_loss_idx                 = np.argsort(holdout_mse_losses)
                self.elite_model_idxes          = sorted_loss_idx[:self.elite_size].tolist()
                # check if any model improves
                break_train = self._save_best(epoch, holdout_mse_losses)
                if break_train:
                    break
                    
        self._track_head_loss(all_holdout_losses)

    def _save_best(
        self, 
        epoch:          int, 
        holdout_losses: torch.tensor    # [ensemble_size]
    ):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                updated = True

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def _track_head_loss(
        self,
        holdout_losses:  List    # [<= max_train_epoch]
    )   -> None:
        self._current_mean_ensemble_losses = np.mean(holdout_losses)


    def predict(
        self,
        inputs:                 torch.Tensor, 
        batch_size:             float   = 1024, 
        factor_ensemble:        bool    = True
    ):

        if inputs.ndim == 2:
            B       = inputs.shape[0]
            inputs  = self.scaler.transform(inputs)
            ensemble_mean, ensemble_var = [], []
            for i in range(0, B, batch_size):
                input = inputs[i:min(i + batch_size, B)]
                b_mean, b_var = self.ensemble_model(
                    input[None, :, :].repeat([self.network_size, 1, 1]), 
                    ret_log_var=False
                )
                ensemble_mean.append(b_mean)
                ensemble_var.append(b_var)
            ensemble_mean   = torch.concat(ensemble_mean, 1)    # concat along the batch_size axis
            ensemble_var    = torch.concat(ensemble_var, 1)

            if factor_ensemble:
                return ensemble_mean, ensemble_var              # [ensemble_size, batch_size, |S|+1]
            else:
                mean    = torch.mean(ensemble_mean, dim=0)
                var     = torch.mean(ensemble_var, dim=0) + torch.mean(torch.square(ensemble_mean - mean[None, :, :]), dim=0)
                return mean, var
        elif inputs.ndim == 3:
            assert inputs.shape[0] == self.network_size
            B       = inputs.shape[1]
            inputs  = self.scaler.transform(inputs)
            ensemble_mean, ensemble_var = [], []
            for i in range(0, B, batch_size):
                input = inputs[:, i:min(i + batch_size, B), :]
                b_mean, b_var = self.ensemble_model(
                    input,
                    ret_log_var=False
                )
                ensemble_mean.append(b_mean)
                ensemble_var.append(b_var)
            ensemble_mean   = torch.concat(ensemble_mean, 1)    # concat along the batch_size axis
            ensemble_var    = torch.concat(ensemble_var, 1)

            if factor_ensemble:
                return ensemble_mean, ensemble_var              # [ensemble_size, batch_size, |S|+1]
            else:
                mean    = torch.mean(ensemble_mean, dim=0)
                var     = torch.mean(ensemble_var, dim=0) + torch.mean(torch.square(ensemble_mean - mean[None, :, :]), dim=0)
                return mean, var
        else:
            raise ValueError            

