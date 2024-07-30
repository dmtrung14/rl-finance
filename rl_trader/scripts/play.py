import sys
sys.path.append('..')
sys.path.append('../..')
from rl_trader import TRADER_ROOT_DIR
import os

import yfinance as yf
from rl_trader.srcs import *
from utils.helpers import welcome, get_args, task_registry

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device='cuda' if torch.cuda.is_available() else 'cpu')
    

    trader_index = 0 # which robot is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards

    # initialize an array of total portfolio values

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        # TODO: calculate portfolio value from obs

        
        # TODO: append portfolio value to the array

    
    # TODO: plot the portfolio values


        

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    welcome()
    args = get_args()
    play(args)