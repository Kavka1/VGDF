from typing import List, Dict, Tuple
import torch
import torch.nn as nn

from VGDF.model.base import call_mlp, Module


class VFunction(Module):
    def __init__(
        self, 
        s_dim: int, 
        hidden_layers: List[int],
        inner_nonlinear: str,
        initializer: str
    ) -> None:
        super(VFunction, self).__init__()
        self.s_dim = s_dim
        self.hidden_layers = hidden_layers

        self.model = call_mlp(
            in_dim              =   s_dim,
            out_dim             =   1,
            hidden_layers       =   hidden_layers,
            inner_activation    = inner_nonlinear,
            output_activation   =   'Identity',
            initializer         =   initializer
        )

    def __call__(self, s: torch.tensor) -> torch.tensor:
        return self.model(s)


class QFunction(Module):
    def __init__(self, s_dim: int, a_dim: int, hidden_layers: List[int], inner_nonlinear: str, initializer: str) -> None:
        super(QFunction, self).__init__()

        self.s_dim, self.a_dim = s_dim, a_dim
        self._model = call_mlp(
            in_dim              =   s_dim + a_dim,
            out_dim             =   1,
            hidden_layers       =   hidden_layers,
            inner_activation    =   inner_nonlinear,
            output_activation   =   'Identity',
            initializer         =   initializer
        )
    
    def forward(self, state: torch.tensor, action: torch.tensor) -> torch.tensor:
        return self._model(torch.concat([state, action], -1))


class QEnsemble(Module):
    def __init__(
        self,
        ensemble_size: int,
        s_dim: int,
        a_dim: int,
        hiddens: List[int],
        inner_nonlinear: str,
        initializer: str
    ) -> None:
        super().__init__()
        self.ensemble_size  = ensemble_size
        self.s_dim          = s_dim 
        self.a_dim          = a_dim
        self.ensemble       = nn.ModuleList(
            [QFunction(s_dim, a_dim, hiddens, inner_nonlinear, initializer) for _ in range(ensemble_size)]
        )

    def forward(self, state: torch.tensor, action: torch.tensor) -> Tuple:
        all_q_values = [qfunction(state, action) for qfunction in self.ensemble]
        return tuple(all_q_values)

    def forward_single(self, state: torch.tensor, action: torch.tensor, index: int) -> torch.tensor:
        return self.ensemble[index](state, action)