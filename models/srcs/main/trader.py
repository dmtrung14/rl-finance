import numpy as np
import pandas as pd
import torch
from .trader_config import TraderCfg
from collections import defaultdict
from utils.stock_groups import Tickers
import yfinance as yf



class Trader():
    def __init__(self, cfg: TraderCfg):
        self.cfg = cfg
        self.max_position = cfg.trader.max_position
        self.close = cfg.trader.close
        self.fee = cfg.market.fee
        self.partial_exchange = cfg.market.partial_exchange
        self.date = pd.to_datetime(cfg.market.start_date)
        self.end = pd.to_datetime(cfg.market.end_date)
        self.finance_df = yf.download(tickers=self.symbols, 
                                      start=self.date - pd.DateOffset(day=14), 
                                      end=self.end + pd.DateOffset(day=14)
                                      ).stack().iloc[:, np.r_[0, 2:6]]
        self._prepare_reward_function()
        self._get_symbols()

        # engine params
        self.num_envs = cfg.envs.num_envs
        self.num_obs = cfg.envs.num_obs
        self.num_privileged_obs = cfg.envs.num_privileged_obs
        self.num_actions = self.num_stocks = len(self.symbols)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # init buffers
        self._init_buffers()
            
    def _init_buffers(self):
        self.pos_buf = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float)
        self.balance_buf = torch.full((self.num_envs,), self.cfg.trader.balance, device=self.device, dtype=torch.float).T
        self.value_buf = self.balance_buf.clone()
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float) if self.num_privileged_obs else None
        self.extras = {}


    def _get_symbols(self):
        self.symbols = []
        for group in self.cfg.market['stock_groups']:
            if group == "tech":
                self.symbols += Tickers.tech
            elif group == "finance":
                self.symbols += Tickers.finance
            elif group == "energy":
                self.symbols += Tickers.energy
            else:
                raise ValueError("Invalid stock group, choose from ['tech', 'finance', 'energy']")

        
    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, which will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
        
    def step(self, actions):
        # actions: tensor of shape (num_envs, num_actions)
        # actions[i, j] is the amount of stock j to buy/sell in env i
        today_close_price = torch.tensor(self.finance_df.loc[self.date].iloc[:,0].values, device=self.device, dtype=torch.float)
        today_open_price = torch.tensor(self.finance_df.loc[self.date].iloc[:,3].values, device=self.device, dtype=torch.float)
        self.pos_buf += torch.matmul(actions, today_close_price)
        self.balance_buf -= torch.matmul(actions, today_open_price) + actions.abs().sum(dim=1) * self.fee
       
        # compute portfolio value
        self.compute_value()

        # cleaning up after stepping
        self.post_step()

        # return observations, rewards, resets, extras
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_step(self):
        self.date += pd.DateOffset(days=1)
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observation()
    
    def compute_value(self):
        self.value_buf = self.balance + sum([amount * stock.price for stock, amount in self.position.items()])

    def check_termination(self):
        self.reset_close_buf = torch.tensor([1 if value < self.close else 0 for value in self.value_buf], device=self.device, dtype=torch.float)
        self.reset_balance_buf = torch.tensor([1 if balance < 0 else 0 for balance in self.balance_buf], device=self.device, dtype=torch.float)
        self.reset_buf = self.reset_close_buf | self.reset_balance_buf
        if self.date >= self.end:
            self.timeout = 1.
            self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.float)
        else: self.timeout = 0.

    def reset_idx(self, idx):
        if len(idx) == self.num_envs:
            self.reset()
            return
        self.pos_buf[idx] = torch.zeros(self.num_actions, device=self.device, dtype=torch.float)
        self.balance_buf[idx] = self.cfg.trader.balance
        self.value_buf[idx] = self.balance_buf[idx]


    def reset(self):
        self.pos_buf[:] = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float)
        self.balance_buf[:] = self.cfg.trader.balance
        self.value_buf[:] = self.balance_buf
        self.date = self.cfg.market.start_date

    def compute_observation(self):
        """
        Computing both observation and privileged observation
        """
        self.prepare_prices()

        self.obs_buf = torch.cat(
            (self.balance_buf,
             self.pos_buf,
             self.price_buf),
            dim=1
        )
        
        self.privileged_obs_buf = torch.cat(
            (self.balance_buf,
             self.pos_buf,
             self.privileged_price_buf),            
            dim=1
        )

    def prepare_prices(self):
        end_date_offset = self.date + pd.DateOffeset(day=1)
        start_date_offset = self.date - pd.DateOffset(day=14)
        last_14_days = self.finance_df.loc[start_date_offset:end_date_offset]
        df_tensor = torch.tensor(last_14_days.values)
        last_5_days = df_tensor[-self.num_stocks * 6:-self.num_stocks].flatten()
        open_price_today = df_tensor[-self.num_stocks:][:, 3].flatten()
        price_today = df_tensor[-self.num_stocks:].flatten()
        obs = torch.cat((last_5_days, open_price_today))
        privileged_obs = torch.cat((last_5_days, price_today))
        self.price_buf = obs.repeat(self.num_envs, 1)
        self.privileged_price_buf = privileged_obs.repeat(self.num_envs, 1)

    def get_observation(self):
        return self.obs_buf
            
    def get_privileged_observation(self):
        return self.privileged_obs_buf

    #------------ reward functions----------------

    def _reward_termination(self):
        return self.reset_buf * ~self.timeout

    def _reward_profit(self):
        return (self.value_buf - self.cfg.trader.balance) / self.cfg.trader.balance

    def _reward_extreme_position(self):
        return torch.clip(self.pos_buf - self.max_position, min = 0)

    def _reward_sharpe(self):
        return 0

    

    