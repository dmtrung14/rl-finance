from rl_trader import TRADER_ROOT_DIR, TRADER_SRCS_DIR
from .main.trader_config import TraderCfg, TraderCfgPPO
from .main.trader import Trader
import os

from rl_trader.utils.task_registry import task_registry

task_registry.register('trader', Trader, TraderCfg(), TraderCfgPPO())