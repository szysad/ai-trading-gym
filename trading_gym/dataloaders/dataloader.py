import abc
from curses import intrflush
from typing import Tuple
from gym.spaces.space import Space
import numpy as np

class Dataloader:
    """Base class for trading env dataloader"""

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __next__(self) -> Space:
        raise NotImplementedError

    @abc.abstractmethod
    def observation_space(self) -> Space:
        raise NotImplementedError
    
    @abc.abstractmethod
    def tradable_items(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def prices(self, obs: Space) -> np.ndarray:
        """Returns table of shape (N, 2) for given observation where
        for each id of tradable item id (from 0 to N-1)
        there is buy and sell price.
        """
        raise NotImplementedError