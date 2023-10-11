from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal

from VGDF.model.base import call_mlp, Module


class FixStdGaussianPolicy(Module):
    def __init__(
        self,
        s_dim: int,
        a_dim: int,
        hidden_layers: List[int],
        inner_nonlinear: str,
        out_nonlinear: str,
        action_std: float,
        initializer: str
    ) -> None:
        super(FixStdGaussianPolicy, self).__init__()
        self.s_dim      = s_dim
        self.a_dim      = a_dim
        self.ac_std     = nn.Parameter(torch.ones(size=(a_dim,)) * action_std, requires_grad=False)

        self._model = call_mlp(
            in_dim              =   s_dim,
            out_dim             =   a_dim,
            hidden_layers       =   hidden_layers,
            inner_activation    =   inner_nonlinear,
            output_activation   =   out_nonlinear,
            initializer         =   initializer
        )

    def sample_action(self, state: torch.tensor, with_noise: bool) -> torch.tensor:
        a_mean = self._model(state)
        if with_noise:
            a_dist = Normal(a_mean, self.ac_std)
            action = a_dist.sample()
            return action
        else:
            return a_mean

    def forward(self, state: torch.tensor) -> torch.distributions:
        a_mean = self._model(state)
        dist   = Normal(a_mean, self.ac_std)
        return dist


class GaussianPolicy(Module):
    def __init__(
        self,
        s_dim: int,
        a_dim: int,
        hidden_layers: List[int],
        inner_nonlinear: str,
        out_nonlinear: str,
        log_std_min: float,
        log_std_max: float,
        initializer: str
    ) -> None:
        super(GaussianPolicy, self).__init__()
        self.s_dim      = s_dim
        self.a_dim      = a_dim

        self._model = call_mlp(
            in_dim              =   s_dim,
            out_dim             =   a_dim,
            hidden_layers       =   hidden_layers,
            inner_activation    =   inner_nonlinear,
            output_activation   =   out_nonlinear,
            initializer         =   initializer
        )

        self.log_std        = nn.Parameter(torch.zeros(size=(a_dim,)))
        self.log_std_min    = log_std_min
        self.log_std_max    = log_std_max
        
    def sample_action(self, state: torch.tensor, with_noise: bool) -> torch.tensor:
        a_mean = self._model(state)
        if with_noise:
            a_std       = torch.exp(self.log_std.clamp(self.log_std_min, self.log_std_max))
            scale_tril  = torch.diag(a_std)
            a_dist      = MultivariateNormal(a_mean, scale_tril=scale_tril)
            action      = a_dist.sample()
            return action
        else:
            return a_mean

    def forward(self, state: torch.tensor) -> torch.distributions:
        a_mean      = self._model(state)
        a_std       = torch.exp(self.log_std.clamp(self.log_std_min, self.log_std_max))
        scale_tril  = torch.diag(a_std)
        a_dist      = MultivariateNormal(a_mean, scale_tril=scale_tril)
        return a_dist

        
class DeterministicPolicy(Module):
    def __init__(
        self,
        s_dim: int,
        a_dim: int,
        hidden_layers: List[int],
        inner_nonlinear: str,
        initializer: str
    ) -> None:
        super(DeterministicPolicy, self).__init__()
        self.s_dim      = s_dim
        self.a_dim      = a_dim

        self._model = call_mlp(
            in_dim              =   s_dim,
            out_dim             =   a_dim,
            hidden_layers       =   hidden_layers,
            inner_activation    =   inner_nonlinear,
            output_activation   =   'Tanh',
            initializer         =   initializer
        )

    def sample_action(self, state: torch.tensor) -> torch.tensor:
        return self._model(state)

    def forward(self, state: torch.tensor) -> torch.tensor:
        return self._model(state)


class SquashedGaussianPolicy(Module):
    def __init__(
        self,
        s_dim: int,
        a_dim: int,
        hidden_layers: List[int],
        inner_nonlinear: str,
        log_std_min: float,
        log_std_max: float,
        initializer: str
    ) -> None:
        super().__init__()
        self.s_dim, self.a_dim = s_dim, a_dim
        self._model = call_mlp(
            s_dim,
            a_dim * 2,
            hidden_layers,
            inner_nonlinear,
            'Identity',
            initializer
        )
        self.log_std_min = nn.Parameter(torch.ones([a_dim]) * log_std_min, requires_grad=False)
        self.log_std_max = nn.Parameter(torch.ones([a_dim]) * log_std_max, requires_grad=False)

    def sample_action(self, state: torch.tensor, with_noise: bool) -> torch.tensor:
        with torch.no_grad():
            mix = self._model(state)
            mean, log_std = torch.chunk(mix, 2, dim=-1)
        if with_noise:
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            std = torch.exp(log_std)
            dist = Normal(mean, std)
            action = dist.sample()
        else:
            action = mean
        return torch.tanh(action)

    def forward(self, state: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.distributions.Distribution]:
        mix             =   self._model(state)
        mean, log_std   =   torch.chunk(mix, 2, dim=-1)
        log_std         =   torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std             =   torch.exp(log_std)

        dist                =   Normal(mean, std)
        arctanh_actions     =   dist.rsample()
        log_prob            =   dist.log_prob(arctanh_actions).sum(-1, keepdim=True)

        action              =   torch.tanh(arctanh_actions)
        squashed_correction =   torch.log(1 - action**2 + 1e-6).sum(-1, keepdim=True)
        log_prob            =   log_prob - squashed_correction

        return action, log_prob, dist