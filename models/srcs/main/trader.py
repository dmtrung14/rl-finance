import torch
from .trader_config import TraderCfg
from collections import defaultdict
import yfinance as yf


class Trader():
    def __init__(self, cfg: TraderCfg):
        self.cfg = cfg
        self.max_position = cfg.trader.max_position
        self.balance = cfg.trader.balance
        self.close = cfg.trader.close
        self.value = cfg.trader.balance

        self.position = defaultdict(int)
        self.terminated = False
        
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

    def compute_value(self):
        self.value = self.balance + sum([amount * stock.price for stock, amount in self.position.items()])
        return self.value
    
    def trade(self, actions):
        date = actions["date"]
        


    def _reset(self):
        self.position = defaultdict(int)
        self.balance = self.cfg.trader.balance
        self.value = self.cfg.trader.balance
        self.terminated = False

    #------------ reward functions----------------

    def _reward_termination(self):
        return int(self.terminated) * 100

    def _reward_profit(self):
        pass

    def _reward_extreme_position(self):
        exceeding_positions = torch.Tensor([amount for amount in self.position.values() if amount > self.max_position])
        return exceeding_positions.sum()/len(exceeding_positions) if len(exceeding_positions) > 0 else 0

    def _reward_sharpe(self):
        pass



    # TODO: Think of other reward functions

    

    