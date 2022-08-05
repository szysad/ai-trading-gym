"""Implements trading env with discrete actions"""
from __future__ import annotations
import gym
from gym import Space, spaces
from gym.core import ObsType, ActType
from enum import Enum
from trading_gym.dataloaders.dataloader import Dataloader
from typing import List, Tuple, Optional, Union, Dict
import numpy as np


class PricesIdx(Enum):
    SELL = 1
    BUY = 0


class DGym(gym.Env):

    metadata = {'render.modes': ['human']}
    action_space: spaces.MultiDiscrete
    observation_space: spaces.Space[ObsType]
    _tick: int
    _end_tick: int
    _done: bool
    _history: List[Tuple[int, ...]]
    _positions: np.ndarray
    _curr_portfolio_value: float
    _curr_prices: np.ndarray

    def __init__(self, dataloader: Dataloader, init_balance: float = 10000, max_trade_quantity: int = 5) -> DGym:
        self.dataloader = dataloader
        self.init_balance = init_balance
        self.max_trade_quantity = max_trade_quantity

        # spaces
        self.action_space = spaces.MultiDiscrete([max_trade_quantity * 2 + 1] * dataloader.tradable_items())
        self.observation_space = dataloader.observation_space()

        # episode
        self._init_params()
        super().__init__()
    
    def _next_obs(self) -> Space:
        if self._tick == len(self.dataloader):
            self._tick = 0
            self._done = True
        else:
            self._tick += 1
        return self.dataloader[self._tick]
        
    def _init_params(self):
        self._tick = 0
        self._end_tick = len(self.dataloader)
        self._done = False
        self._balance = self.init_balance
        self._history = []
        self._positions = np.zeros(self.dataloader.tradable_items())
        self._curr_portfolio_value = self._balance
    
    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> Union[ObsType, Tuple[ObsType, dict]]:
        self._init_params()
        obs = self._next_obs()
        self._curr_prices = self.dataloader.prices(obs)
        assert self._tick < len(self.dataloader), "Episode must not end immidiately after reset"
        return obs
    
    def _try_performing_action(self, action: spaces.MultiDiscrete, prices: np.ndarray):
        order = action - self.max_trade_quantity
        sell_ammounts = np.where(action < 0, -order, 0)
        buy_ammounts = np.where(action > 0, order, 0)
        buy_price = np.sum(prices[buy_ammounts])

        if np.all(self._positions >= sell_ammounts) and buy_price <= self._balance:
            # all orders can be performed
            self._positions = self._positions - sell_ammounts + buy_ammounts
            self._balance += np.sum(prices[order])
        else:
            # illegal action
            pass
    
    def _calculate_porfolio_value(self, prices: np.ndarray) -> float:
        return self._balance + np.sum(prices[self._positions])

    def _calculate_reward(self, last_portfolio_value: float, new_portfolio_value: float) -> float:
        return new_portfolio_value - last_portfolio_value
    
    def step(self, action: ActType) -> Union[Tuple[ObsType, float, bool, bool, dict], Tuple[ObsType, float, bool, dict]]:
        self._try_performing_action(action=action, prices=self._curr_prices)
        new_portfolio_value = self._calculate_porfolio_value(self._curr_prices)
        reward = self._calculate_reward(self._curr_prices, new_portfolio_value)

        observation = self._next_obs()
        self._curr_portfolio_value = new_portfolio_value
        self._curr_prices = self.dataloader.prices(observation)
        return observation, reward, self._done, {}
