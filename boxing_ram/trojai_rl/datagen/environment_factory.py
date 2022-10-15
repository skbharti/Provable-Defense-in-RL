import logging
from abc import ABC, abstractmethod

import gym

logger = logging.getLogger(__name__)


class EnvironmentFactory(ABC):
    """ Factory object that returns RL environments for training. """
    @abstractmethod
    def new_environment(self, *args, **kwargs) -> gym.Env:
        """
        Returns a new Trojai RL environment
        """
        pass

    def __eq__(self, other):
        pass
