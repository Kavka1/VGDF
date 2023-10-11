from typing import Dict, List
import random
from pathlib import Path
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
import yaml

from VGDF.env.env_utils     import call_terminal_func
from VGDF.env.common        import call_env
from VGDF.env.sampler       import EnvSampler
from VGDF.misc.utils        import seed_all, make_exp_path
from VGDF.misc.logger       import Logger
from VGDF.agent.vgdf        import VGDF_Agent


def train(config: Dict, exp_name: str = None) -> False:
    result_path             =   f"{str(Path(__file__).parent.absolute())}/../results/vgdf/"
    config['result_path']   =   result_path

    seed_all(config['seed'])
    # train env
    src_env     = call_env(config['src_env_config'])
    src_env.seed(config['seed'])
    src_sampler = EnvSampler(src_env)
    # test env
    tar_env     = call_env(config['tar_env_config'])
    tar_env.seed(config['seed'])
    tar_sampler = EnvSampler(tar_env)
    # update config
    config['model_config'].update({
        's_dim':                src_env.observation_space.shape[0],
        'a_dim':                src_env.action_space.shape[0],
    })
    make_exp_path(config, exp_name)
    # logger
    tb      = SummaryWriter(config['exp_path'])
    logger  = Logger()
    # agent
    agent = VGDF_Agent(config)
    # start rollout
    total_step, total_episode, best_train_score, best_test_score = 0, 0, 0, 0
    total_step_in_tar_env = 0
    while total_step <= config['max_step']:
        if total_step % config['eval_freq'] == 0:
            # obtain the loss log
            temp_log = agent.loss_log
            temp_log.update({'step': total_step})
            # evaluate in src env
            all_scores = src_sampler.evaluate(agent, config['eval_episode'], return_full=True)
            all_scores = np.sort(all_scores)
            src_mean_score = np.mean(all_scores)
            src_cvar_score = np.mean(all_scores[:max(1, int(len(all_scores)*0.25))])
            temp_log[f'src env avg score'] = src_mean_score 
            temp_log[f'src env cvar score'] = src_cvar_score 
            # evaluate in tar env
            all_scores = tar_sampler.evaluate(agent, config['eval_episode'], return_full=True, render=False)
            all_scores = np.sort(all_scores)
            tar_mean_score = np.mean(all_scores)
            tar_cvar_score = np.mean(all_scores[:max(1, int(len(all_scores)*0.25))])
            temp_log[f'tar env avg score'] = tar_mean_score 
            temp_log[f'tar env cvar score'] = tar_cvar_score 
            # log
            temp_log['step in target']  =   total_step_in_tar_env
            logger.store(**temp_log)
            logger.print_all()
            # tb
            for key, val in list(temp_log.items()):
                tb.add_scalar(f'train/{key}', val, total_step)
            tb.add_scalar(f'eval/tar_env_score_wrt_tar_step', tar_mean_score, total_step_in_tar_env)
            # save best model
            if src_mean_score > best_train_score:
                agent.save_all_module(f"best_scr")
                best_train_score    =   src_mean_score
            if tar_mean_score > best_test_score:
                agent.save_all_module(f'best_tar')
                best_test_score     =   tar_mean_score
        # save model
        if total_step % config['save_freq'] == 0:
            agent.save_all_module(f"{total_step}")
            logger.save(config['exp_path'] + 'log.pkl')

        # interaction with src env
        if config['no_optimistic']:
            cur_state, action, reward, done, next_state, info = src_sampler.sample(agent, with_noise=True, optimistic=False)
        else:
            cur_state, action, reward, done, next_state, info = src_sampler.sample(agent, with_noise=True, optimistic=True)
    
        agent.src_buffer.store((cur_state, action, [reward], [done], next_state))
        # interaction with tar env
        if total_step % config['tar_env_interact_freq'] == 0:
            tar_s, tar_a, tar_r, tar_done, tar_next_s, _  = tar_sampler.sample(agent=agent, with_noise=True)
            agent.tar_buffer.store((tar_s, tar_a, [tar_r], [tar_done], tar_next_s))
            total_step_in_tar_env += 1
        # train model
        if total_step % config['dynamics_train_freq'] == 0:
            agent.train_model(total_step)
        # train ac
        agent.train_ac(
            total_step, 
            call_terminal_func(config['src_env_config']['env_name']),
        )

        total_step += 1

    agent.save_all_module('final')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",              help="set (multiple) random seed", type=int, nargs='+', default=[10])
    parser.add_argument('--exp_name',           help="describe the experiment", type=str, default=None)
    parser.add_argument('--env',                help='choose the environment', type=str, default='walker')
    parser.add_argument('--no_optimistic',      help='whether use optimistic sampling', action='store_true')
    parser.add_argument('--data_sharing_ratio', help='utilized ratio from the source domain', default=0.75, type=float)
    parser.add_argument('--ensemble_size',      help='dynamics ensemble size', default=7, type=int)    
    args = parser.parse_args()

    with open(f"{str(Path(__file__).parent.absolute())}/../config/vgdf/{args.env}.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.no_optimistic:
        config['no_optimistic'] = True
    else:
        config['no_optimistic'] = False
    
    config['likelihood_gate_threshold']                 = args.data_sharing_ratio
    config['model_config']['dynamics_ensemble_size']    = args.ensemble_size
    config['model_config']['dynamics_elite_size']       = max(args.ensemble_size - 2, 1)

    for seed in args.seeds:
        config['seed'] = seed
        train(config, args.exp_name)