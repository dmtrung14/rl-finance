from .main.trader_config import TraderCfg, TraderCfgPPO
from .main.trader import Trader

from rl_trader.utils.task_registry import task_registry

task_registry.register('trader', Trader, TraderCfg(), TraderCfgPPO())