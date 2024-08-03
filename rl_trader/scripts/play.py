import sys
sys.path.append('..')
sys.path.append('../..')

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from utils.helpers import welcome, get_args
from utils.task_registry import task_registry
from rl_trader.srcs import *

import torch
import pandas as pd

num_days = 365
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
    
    # initialize an array of total portfolio values
    portfolio_values = []
    for i in range(num_days):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        # TODO: calculate portfolio value from obs
        portfolio_value = torch.mean(obs[:, 13])
        portfolio_values.append(portfolio_value)
    
    # TODO: plot the portfolio values
    plot(portfolio_values)

    

def plot(portfolio_values):
    # convert normalized_values from tensor to numerical
    portfolio_values = [value.item() for value in portfolio_values]

    # create a list of dates of num_dates up to end_date
    end_date = pd.Timestamp('2024-05-01')
    dates = pd.date_range(start=end_date - pd.DateOffset(days=num_days-1), end=end_date, freq='D')

    # create a DataFrame with dates as index and portfolio_values as the only column
    portfolio_values_df = pd.DataFrame(portfolio_values, index=dates, columns=['Portfolio Value'])

    plt.style.use('ggplot')
    portfolio_values_df.plot(figsize=(10, 5))
    
    plt.show()


        

if __name__ == '__main__':
    welcome()
    args = get_args()
    play(args)