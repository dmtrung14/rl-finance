import sys
sys.path.append('..')
sys.path.append('../..')

import numpy as np
from datetime import datetime
from utils.helpers import set_seed, get_args, welcome
from utils.task_registry import task_registry
from rl_trader.srcs import *

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    welcome()
    args = get_args()
    train(args)