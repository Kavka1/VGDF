from typing import Callable, Dict, Tuple, List, Union
import gym
import numpy as np
import copy
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import d4rl

from VGDF.misc.buffer                   import Buffer
from VGDF.misc.utils                    import soft_update, confirm_path_exist
from VGDF.model.policy                  import SquashedGaussianPolicy
from VGDF.model.value                   import QEnsemble
from VGDF.model.dynamics                import EnsembleDynamicsModel


class VGDF_BC_Agent:
    def __init__(self, config: Dict) -> None:
        self.config             =       config
        self.model_config       =       config['model_config']
        self.s_dim              =       self.model_config['s_dim']
        self.a_dim              =       self.model_config['a_dim']
        self.device             =       config['device']
        self.exp_path           =       config['exp_path']

        self.lr                 =       config['lr']
        self.gamma              =       config['gamma']
        self.tau                =       config['tau']
        self.alpha              =       config['alpha']
        self.training_delay     =       config['training_delay']

        self.batch_size         =       config['batch_size']
        
        self.ac_gradient_clip   =       config['ac_gradient_clip']

        self.dynamics_batch_size                    =       config['dynamics_batch_size']
        self.dynamics_holdout_ratio                 =       config['dynamics_holdout_ratio']
        self.dynamics_max_epochs_since_update       =       config['dynamics_max_epochs_since_update']
        self.max_epochs_since_update_decay_interval =       config['max_epochs_since_update_decay_interval']
        
        self.start_gate_src_sample          =       config['start_gate_src_sample']
        self.likelihood_gate_threshold      =       config['likelihood_gate_threshold']
        
        # bc 
        self.bc_coeff                       =       config['bc_coeff']
        self.use_q_decay                    =       config['use_q_decay']

        self.training_count =       0
        self.loss_log       =       {}

        # adaptive alpha
        if self.alpha is None:
            self.train_alpha    =   True
            self.log_alpha      =   torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha          =   torch.exp(self.log_alpha)
            self.target_entropy =   - torch.tensor(config['model_config']['a_dim'], dtype=torch.float64)
            self.optimizer_alpha=   optim.Adam([self.log_alpha], lr= self.lr)
        else:
            self.train_alpha    =   False

        # policy
        self.policy         =       SquashedGaussianPolicy(
            s_dim           =   self.s_dim,
            a_dim           =   self.a_dim,
            hidden_layers   =   self.model_config['policy_hiddens'],
            inner_nonlinear =   self.model_config['policy_nonlinear'],
            log_std_min     =   self.model_config['policy_log_std_min'],
            log_std_max     =   self.model_config['policy_log_std_max'],
            initializer     =   self.model_config['policy_initializer']
        ).to(self.device)
        self.optimizer_policy   =   optim.Adam(self.policy.parameters(), self.lr)
        
        # value functions
        self.QFunction      =       QEnsemble(
            ensemble_size   =   2,
            s_dim           =   self.s_dim,
            a_dim           =   self.a_dim,
            hiddens         =   self.model_config['value_hiddens'],
            inner_nonlinear =   self.model_config['value_nonlinear'],
            initializer     =   self.model_config['value_initializer']
        ).to(self.device)
        self.QFunction_tar  =      QEnsemble(
            ensemble_size   =   2,
            s_dim           =   self.s_dim,
            a_dim           =   self.a_dim,
            hiddens         =   self.model_config['value_hiddens'],
            inner_nonlinear =   self.model_config['value_nonlinear'],
            initializer     =   self.model_config['value_initializer']
        ).to(self.device)
        self.QFunction_tar.load_state_dict(self.QFunction.state_dict())
        self.optimizer_value    =   optim.Adam(self.QFunction.parameters(), self.lr)

        # batch dynamics model
        self.dynamics   =   EnsembleDynamicsModel( 
            network_size=   self.model_config['dynamics_ensemble_size'],
            elite_size  =   self.model_config['dynamics_elite_size'],
            state_size  =   self.s_dim,
            action_size =   self.a_dim,
            reward_size =   1,
            hidden_size =   self.model_config['dynamics_hidden_size'],
            use_decay   =   True,
            device      =   config['device']
        ).to(self.device)

        self.src_buffer         =   Buffer(config['src_buffer_size'])
        self.tar_buffer         =   Buffer(config['tar_buffer_size'])

        self._load_d4rl_2_buffer(config['offline_dataset'])

    def _load_d4rl_2_buffer(self, d4rl_offline_mark: str) -> None:
        env     = gym.make(d4rl_offline_mark)
        dataset = env.get_dataset()
        # 2 buffer
        s, a, r, done, next_s = dataset['observations'], dataset['actions'], dataset['rewards'], dataset['terminals'], dataset['next_observations']
        for idx in range(s.shape[0]):
            self.src_buffer.store((s[idx], a[idx], [r[idx]], [done[idx]], next_s[idx]))
        # log
        print(f"Complete offline dataset {d4rl_offline_mark} transfer")

    def sample_action(self, s: np.array, with_noise: bool) -> np.array:
        with torch.no_grad():
            s               =   torch.from_numpy(s).float().to(self.device)
            action          =   self.policy.sample_action(s, with_noise)
        return action.detach().cpu().numpy()

    def train_model(self, current_step: int,) -> None:
        if len(self.src_buffer) < self.dynamics_batch_size:
            return

        # decay the max epochs since update coefficient
        current_dynamics_max_epochs_since_update    =   max(
            0,
            self.dynamics_max_epochs_since_update - int(current_step / self.max_epochs_since_update_decay_interval)
        )

        s_batch, a_batch, r_batch, done_batch, next_s_batch = self.tar_buffer.sample_all()
        delta_s_batch = next_s_batch - s_batch
        inputs      = np.concatenate((s_batch, a_batch), axis=-1)
        labels      = np.concatenate((r_batch, delta_s_batch), axis=-1)
        self.dynamics.train(
            inputs                  =   inputs,
            labels                  =   labels,
            batch_size              =   self.dynamics_batch_size,
            holdout_ratio           =   self.dynamics_holdout_ratio,
            max_epochs_since_update =   current_dynamics_max_epochs_since_update
        )
        self.loss_log[f'loss_dynamics']                             =   self.dynamics._current_mean_ensemble_losses
        self.loss_log['current_dynamics_max_epochs_since_update']   =   current_dynamics_max_epochs_since_update

    def train_ac(self, current_step: int, terminal_func: Callable = None) -> None:
        if len(self.src_buffer) < self.batch_size or len(self.tar_buffer) < self.batch_size:
            return

        src_s, src_a, src_r, src_done, src_next_s       =   self.src_buffer.sample(self.batch_size)
        src_s         =       torch.from_numpy(src_s).float().to(self.device)
        src_a         =       torch.from_numpy(src_a).float().to(self.device)
        src_r         =       torch.from_numpy(src_r).float().to(self.device)
        src_done      =       torch.from_numpy(src_done).float().to(self.device)
        src_next_s    =       torch.from_numpy(src_next_s).float().to(self.device)

        tar_s, tar_a, tar_r, tar_done, tar_next_s       =   self.tar_buffer.sample(self.batch_size)
        tar_s         =       torch.from_numpy(tar_s).float().to(self.device)
        tar_a         =       torch.from_numpy(tar_a).float().to(self.device)
        tar_r         =       torch.from_numpy(tar_r).float().to(self.device)
        tar_done      =       torch.from_numpy(tar_done).float().to(self.device)
        tar_next_s    =       torch.from_numpy(tar_next_s).float().to(self.device)
        
        assert len(src_done.shape) == len(src_r.shape) == len(tar_r.shape) == len(tar_done.shape) == 2

        # training value function
        loss_value, chosen_src_sample_idx      =       self._compute_value_loss(
            src_s, src_a, src_r, src_done, src_next_s,
            tar_s, tar_a, tar_r, tar_done, tar_next_s,
            terminal_func,
            current_step
        )
        self.optimizer_value.zero_grad()
        loss_value.backward()
        critic_total_norm = nn.utils.clip_grad_norm_(self.QFunction.parameters(), self.ac_gradient_clip)
        self.optimizer_value.step()
        self.loss_log['loss_value']         = loss_value.cpu().item()
        self.loss_log['value_total_norm']   = critic_total_norm.detach().cpu().item()

        if self.training_count % self.training_delay == 0:
            # train policy
            loss_policy, new_a_log_prob = self._compute_policy_loss(
                src_s   =      src_s,
                src_a   =      src_a,
                concat_s=      torch.concat([src_s, tar_s], 0)
            )
            self.optimizer_policy.zero_grad()
            loss_policy.backward()
            policy_total_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.ac_gradient_clip)
            self.optimizer_policy.step()
            self.loss_log['loss_policy']    = loss_policy.cpu().item()
            self.loss_log['policy_total_norm'] = policy_total_norm.detach().cpu().item()

            if self.train_alpha:
                loss_alpha  =  (- torch.exp(self.log_alpha) * (new_a_log_prob.detach() + self.target_entropy)).mean()
                self.optimizer_alpha.zero_grad()
                loss_alpha.backward()
                self.optimizer_alpha.step()
                self.alpha = torch.exp(self.log_alpha)
                
                self.loss_log['alpha'] = self.alpha.detach().cpu().item()
                self.loss_log['loss_alpha'] = loss_alpha.detach().cpu().item()

        # soft update target networks
        soft_update(self.QFunction, self.QFunction_tar, self.tau)
        self.training_count += 1

    def _compute_value_loss(
        self, 
        src_s: torch.tensor, src_a: torch.tensor, src_r: torch.tensor, src_done: torch.tensor, src_next_s: torch.tensor,
        tar_s: torch.tensor, tar_a: torch.tensor, tar_r: torch.tensor, tar_done: torch.tensor, tar_next_s: torch.tensor,
        terminal_func: Callable,
        current_step: int,
        src_hist_next_a: torch.tensor = None
    ) -> torch.tensor:
        # first calculate the value loss wrt samples from target domain
        with torch.no_grad():
            tar_next_a, tar_next_a_logprob, _   = self.policy(tar_next_s)
            tar_next_sa_q1, tar_next_sa_q2      = self.QFunction_tar(tar_next_s, tar_next_a)
            tar_next_sa_q                       = torch.min(tar_next_sa_q1, tar_next_sa_q2)
            tar_value_target                    = tar_r + (1 - tar_done) * self.gamma * (tar_next_sa_q - self.alpha * tar_next_a_logprob)
        tar_q1, tar_q2  =   self.QFunction(tar_s, tar_a)
        tar_q_loss      =   F.mse_loss(tar_q1, tar_value_target) + F.mse_loss(tar_q2, tar_value_target)

        # then calculate the value loss wrt samples from source domain
        ## 1. obtain the value target of source domain trans
        with torch.no_grad():
            src_next_a, src_next_a_logprob, _   = self.policy(src_next_s)
            src_next_sa_q1, src_next_sa_q2      = self.QFunction_tar(src_next_s, src_next_a)
            src_next_sa_q                       = torch.min(src_next_sa_q1, src_next_sa_q2)
            src_value_target                    = src_r + (1 - src_done) * self.gamma * (src_next_sa_q - self.alpha * src_next_a_logprob)
        ## 2. obtain the src value targets for comparison
            src_value_target_for_compare    = src_r + (1 - src_done) * self.gamma * src_next_sa_q
        ## 3. expand the dynamics model (of target domain) for TD-h value target distribution
            tar_TD_H_value_target               = self._value_expansion(src_s, src_a, terminal_func)            # [ensemble_size, batch_size, 1]
            tar_TD_H_value_mean                 = torch.mean(tar_TD_H_value_target, dim=0, keepdim=False)       # [batch_size, 1]
            tar_TD_H_value_std                  = torch.std(tar_TD_H_value_target, dim=0, keepdim=False)        # [batch_size, 1]
            tar_TD_H_value_dist                 = torch.distributions.Normal(loc=tar_TD_H_value_mean, scale=tar_TD_H_value_std + 1e-8)
            self.loss_log['generated_value_mean'] = tar_TD_H_value_mean.mean().detach().item()
            self.loss_log['generated_value_std']  = tar_TD_H_value_mean.std().detach().item()
            self.loss_log['generated_value_max']  = tar_TD_H_value_mean.max().detach().item()
            self.loss_log['generated_value_min']  = tar_TD_H_value_mean.min().detach().item()
            ## obtain the value difference
            value_difference                        = torch.abs(tar_TD_H_value_mean - src_value_target_for_compare)
            self.loss_log['value_difference_mean']  = value_difference.mean().detach().item()
            self.loss_log['value_difference_std']   = value_difference.std().detach().item()
            self.loss_log['value_difference_max']   = value_difference.max().detach().item()
            self.loss_log['value_difference_min']   = value_difference.min().detach().item()
            ## obtain the likelihood
            src_value_target_in_dist_likelihood = torch.exp(tar_TD_H_value_dist.log_prob(src_value_target_for_compare)) # [batch_size, 1]
            self.loss_log['src_value_target_likelihood_mean']   =   src_value_target_in_dist_likelihood.mean().detach().item()
            self.loss_log['src_value_target_likelihood_std']    =   src_value_target_in_dist_likelihood.std().detach().item()
        # 4. reject sampling the src samples with likelihood under threshold
            if current_step > self.start_gate_src_sample:
                threshold_likelihood    =   torch.quantile(
                    src_value_target_in_dist_likelihood,
                    q   = self.likelihood_gate_threshold,
                )   # []
                accept_gate             =   (src_value_target_in_dist_likelihood > threshold_likelihood)
            else:
                accept_gate             =   torch.ones_like(src_value_target, device=self.device)
            src_chosen_sample_idx       =   torch.where(accept_gate[:, 0] > 0)[0]
            self.loss_log['accept_gated_ratio']    =   torch.sum(accept_gate.int()).detach().item() / np.prod(accept_gate.shape)
        
        # 5. obtain the loss wrt src samples
        src_q1, src_q2  =   self.QFunction(src_s, src_a)
        src_q_loss      =   (accept_gate * (src_q1 - src_value_target) ** 2).mean() + (accept_gate * (src_q2 - src_value_target) ** 2).mean()
        
        self.loss_log['q_loss_src'] = src_q_loss.detach().item()
        self.loss_log['q_loss_tar'] = tar_q_loss.detach().item()
        return tar_q_loss + src_q_loss, src_chosen_sample_idx
    
    def _value_expansion(self, src_s: torch.tensor, src_a: torch.tensor, terminal_func: Callable) -> torch.tensor:
        # imagine the next s in target domain
        dyna_pred_mean, dyna_pred_var = self.dynamics.predict(inputs=torch.concat([src_s, src_a], -1), factor_ensemble=True)   # [ensemble_size, batch_size, 1 + |S|]
        dyna_pred_samples               =   dyna_pred_mean + torch.ones_like(dyna_pred_var, device=self.device) * dyna_pred_var
        dyna_pred_r, dyna_pred_delta_s  =   dyna_pred_samples[:, :, :1], dyna_pred_samples[:, :, 1:]
        dyna_pred_next_s                =   src_s + dyna_pred_delta_s 

        # expand via dynamics model
        state               =   dyna_pred_next_s
        cumulate_r          =   dyna_pred_r
        discount            =   self.gamma
        notdone             =   terminal_func(dyna_pred_next_s, return_done=False)
        
        # value target of imagined target domain transition
        final_pred_next_s   =   state
        final_action        =   self.policy.sample_action(final_pred_next_s, with_noise=False)
        final_q1, final_q2  =   self.QFunction_tar(final_pred_next_s, final_action)
        final_value         =   torch.min(final_q1, final_q2)   # [ensemble_size, batch_size, 1]
    
        TD_H_Value  =   cumulate_r + notdone * discount * final_value
        return TD_H_Value

    def _compute_policy_loss(self, src_s: torch.tensor, src_a: torch.tensor, concat_s: torch.tensor) -> torch.tensor:
        a, a_log_prob, _    =   self.policy(concat_s)
        q1_value, q2_value  =   self.QFunction(concat_s, a)
        q_value             =   torch.min(q1_value, q2_value)
        neg_entropy         =   a_log_prob
        loss_tar            =   (- q_value + self.alpha * neg_entropy).mean()
        # BC loss
        offline_src_a, _, _ =   self.policy(src_s)
        bc_loss             =   F.mse_loss(offline_src_a, src_a)
        if self.use_q_decay:
            lmbda           =   self.bc_coeff / q_value.abs().mean().detach()
        else:
            lmbda           =   self.bc_coeff

        return lmbda * loss_tar + bc_loss, a_log_prob

    def evaluate(self, env, num_episode: int) -> float:
        total_r = []
        for _ in range(num_episode):
            s = env.reset()
            done = False
            episode_r = 0
            while not done:
                a = self.sample_action(s, False)
                s_, r, done, _ = env.step(a)
                episode_r += r
                s = s_
            total_r.append(episode_r)
        return total_r

    def save_all_module(self, remark: str) -> None:
        model_path = self.exp_path + 'model/'
        confirm_path_exist(model_path)
        model_path = model_path + f'{remark}'
        torch.save({
            'policy':           self.policy.state_dict(),
            'value':            self.QFunction.state_dict(),
            'dynamics':         self.dynamics.state_dict()
        }, model_path)
        print(f"------- All modules saved to {model_path} ----------")
    
    def load_all_module(self, checkpoint_path: str) -> None:
        state_dict = torch.load(checkpoint_path)
        self.policy.load_state_dict(state_dict['policy'])
        self.QFunction.load_state_dict(state_dict['value'])
        self.QFunction_tar.load_state_dict(self.QFunction.state_dict())
        self.dynamics.load_state_dict(state_dict['dynamics'])
        print(f"------- Loaded all modules from {checkpoint_path} ----------")