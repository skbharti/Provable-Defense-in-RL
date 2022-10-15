import logging
from abc import ABC, abstractmethod
from typing import Any

from .statistics import TestStatistics, TrainingStatistics
from ..datagen.environment_factory import EnvironmentFactory

logger = logging.getLogger(__name__)


class RLOptimizerInterface(ABC):
    """Object that performs training and testing of TrojAI RL models."""
    @abstractmethod
    def train(self, model: Any, env_factory: EnvironmentFactory) -> (Any, TrainingStatistics):
        """
        Train the given model using parameters in self.training_params
        :param model: (Any) The untrained model
        :param env_factory: (EnvironmentFactory)
        :return: (Any, TrainingStatistics) trained model and TrainingStatistics object
        """
        pass

    @abstractmethod
    def test(self, model: Any, env_factory: EnvironmentFactory) -> TestStatistics:
        """
        Perform whatever tests desired on the model with clean data and triggered data, return a dictionary of results.
        :param model: (Any) Trained model
        :param env_factory: (EnvironmentFactory)
        :return: (Any, TestStatistics) a TestStatistics object
        """
        pass

    @abstractmethod
    def get_device_type(self) -> str:
        """
        Return a string representation of the type of device used by the optimizer to train the model.
        """
        pass

    @abstractmethod
    def get_cfg_as_dict(self) -> dict:
        """
        Return a dictionary with key/value pairs that describe the parameters used to train the model.
        """
        pass

    @abstractmethod
    def __deepcopy__(self, memodict={}):
        """
        Required for training on clusters. Return a deep copy of the optimizer.
        """
        pass

    @abstractmethod
    def __eq__(self, other):
        """
        Required for training on clusters. Define how to check if two optimizers are equal.
        """
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def save(self, fname: str) -> None:
        """
        Save the optimizer to a file
        :param fname - the filename to save the optimizer to
        """
        pass

    @staticmethod
    @abstractmethod
    def load(fname: str):
        """
        Load an optimizer from disk and return it
        :param fname: the filename where the optimizer is serialized
        :return: The loaded optimizer
        """
        pass
