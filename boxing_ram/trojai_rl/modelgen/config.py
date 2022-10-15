import logging
import os
import re
import uuid
from typing import Any

import torch.nn as nn

from .optimizer_interface import RLOptimizerInterface
from ..datagen.environment_factory import EnvironmentFactory
from .utils import is_jsonable

logger = logging.getLogger(__name__)
SUPPORTED_TRAINING_ALGOS = ['ppo']


class RLOptimizerConfig:
    """
    Defines configuration parameters for RL training
    """

    def __init__(self, algorithm: str = 'ppo', num_frames: int = int(8e6),
                 max_num_frames_rollout: int = 128, num_epochs: int = 1000,
                 device: str = 'cuda', num_frames_per_test: int = int(5e5),
                 learning_rate: float = 1e-3):
        self.algorithm = algorithm
        self.num_frames = num_frames
        self.max_num_frames_rollout = max_num_frames_rollout
        self.num_epochs = num_epochs
        self.device = device
        self.num_frames_per_test = num_frames_per_test
        self.learning_rate = learning_rate

        self.validate()

    def validate(self):
        if not isinstance(self.algorithm, str) or self.algorithm not in SUPPORTED_TRAINING_ALGOS:
            msg = "algorithm input must be a string, and one of:" + str(SUPPORTED_TRAINING_ALGOS)
            logger.error(msg)
            raise ValueError(msg)
        if not isinstance(self.num_frames, int) or self.num_frames < 1:
            msg = "num_frames must be at least 1!"
            logger.error(msg)
            raise ValueError(msg)
        if not isinstance(self.max_num_frames_rollout, int) or self.max_num_frames_rollout < 1:
            msg = "max_num_frames_rollout must be an integer > 0"
            logger.error(msg)
            raise ValueError(msg)
        if not isinstance(self.num_epochs, int) or self.num_epochs < 1:
            msg = "num_epochs must be an integer > 0"
            logger.error(msg)
            raise ValueError(msg)
        if not isinstance(self.device, str) or (
                self.device != 'cpu' and self.device != 'cuda' or not re.match(r'cuda:\d', self.device)):
            msg = "device specification must be a string: either cpu, cuda, cuda:#, where # is an integer >= 0"
            logger.error(msg)
            raise ValueError(msg)
        if self.num_frames_per_test is not None and \
                (not isinstance(self.num_frames_per_test, int) or self.num_frames_per_test < 1):
            msg = "num_frames_per_test must be an integer > 1"
            logger.error(msg)
            raise ValueError(msg)
        if not isinstance(self.learning_rate, float) or self.learning_rate <= 0:
            msg = "learning_rate must be a float > 0"
            logger.error(msg)
            raise ValueError(msg)


class RunnerConfig:
    """
    Defines a runner configuration object, required to configure a Runner to train RL models
    """

    def __init__(self, train_env_factory: EnvironmentFactory, test_env_factory: EnvironmentFactory,
                 trainable_model: nn.Module,
                 optimizer: RLOptimizerInterface,
                 parallel: bool = False,
                 model_save_dir: str = "/tmp/models", stats_save_dir: str = "/tmp/model_stats",
                 run_id: Any = None, filename: str = None, save_with_hash: bool = False,
                 save_info: dict = None):
        """
        Initializes the RunnerConfig object
        :param train_env_factory: (EnvironmentFactory) environment factory for producing training environments
        :param test_env_factory: (EnvironmentFactory) similar to train_env_factory, but for the test environments
        :param trainable_model: (nn.Module) model to be trained
        :param optimizer: (RLOptimizerInterface) RLOptimizerInterface object that will be used to train and test the
            model
        :param parallel: (bool) Whether to run training in parallel.
            Note: while currently unused by current optimizers, we expect this to provide additional instruction about
                parallelization to the optimizer if implemented
        :param model_save_dir: (str) path to folder where models should be saved
        :param stats_save_dir: (str) path to folder where train/test stats should be saved
        :param run_id: (int) optional id to use to identify this run
        :param filename: (str) filename under which to save the model and stats
        :param save_with_hash: (bool) save the model and stats under a hash
        :param save_info: (dict) optional dictionary of json serializable information to save with train and test stats
        """
        self.train_env_factory = train_env_factory
        self.test_env_factory = test_env_factory
        self.trainable_model = trainable_model
        self.optimizer = optimizer

        self.parallel = parallel
        self.model_save_dir = model_save_dir
        self.stats_save_dir = stats_save_dir
        self.run_id = run_id

        self.filename = filename
        self.save_with_hash = save_with_hash
        self.save_info = save_info

        self.validate()

        # quick, hack-y way to do this, may need to be updated if doesn't work later; should maybe go in runner?
        if self.save_with_hash:
            self.filename += '.' + str(uuid.uuid1().hex)

    def validate(self):
        if not type(self.model_save_dir) == str:
            msg = "Expected type 'string' for argument 'model_save_dir, instead got type: " \
                  "{}".format(type(self.model_save_dir))
            logger.error(msg)
            raise TypeError(msg)
        if not os.path.isdir(self.model_save_dir):
            try:
                os.makedirs(self.model_save_dir)
            except OSError as e:  # not sure this error is possible as written
                msg = "'model_save_dir' was not found and could not be created" \
                      "...\n{}".format(e.__traceback__)
                logger.error(msg)
                raise OSError(msg)
        if not os.path.isdir(self.stats_save_dir):
            try:
                os.makedirs(self.stats_save_dir)
            except OSError as e:  # not sure this error is possible as written
                msg = "'stats_save_dir' was not found and could not be created" \
                      "...\n{}".format(e.__traceback__)
                logger.error(msg)
                raise OSError(msg)
        if not type(self.filename) == str:
            msg = "Expected a string for argument 'filename', instead got " \
                  "type {}".format(type(self.filename))
            logger.error(msg)
            raise TypeError(msg)
        if not isinstance(self.save_with_hash, bool):
            msg = "Expected boolean for argument save_with_hash"
            logger.error(msg)
            raise TypeError(msg)
        if not isinstance(self.save_info, dict):
            msg = "Expected type 'dict' for argument 'save_info', instead got type {}".format(type(self.save_info))
            logger.error(msg)
            raise TypeError(msg)
        if not is_jsonable(self.save_info):
            msg = "Argument 'save_info', must be json serializable."
            logger.error(msg)
            raise TypeError(msg)
        if not isinstance(self.parallel, bool):
            msg = "Expected boolean for argument 'parallel', instead got type {}".format(type(self.parallel))
            logger.error(msg)
            raise TypeError(msg)


class TestConfig:
    def __init__(self, environment_cfg: Any, count: int = 100, test_description: dict = None,
                 agent_argmax_action: bool = False):
        """
        Test configuration specification for a single run of an agent through an environment.
        :param environment_cfg: (Any) This is whatever should be passed to the environment factory to instantiate
            an environment.
        :param count: (int) Number of episodes to run the agent through the environment. 
        :param test_description: (dict) A dictionary of key, value pairs providing information about the test; currently
             the only required key is 'poison', whose value should be a string describing the poison strategy or 'clean'
             for no poison. This is also used to save data, and should be mutable to include any information desired to
             be used or saved with the test results.
        :param agent_argmax_action: (bool) Have the agent choose the argmax of it policy distribution. torch_ac has
            the model return a distribution from which it samples an action, set this to True to instead choose the
            agent's highest confidence action.
        """
        self.env_cfg = environment_cfg
        self.count = count
        self.desc = test_description
        self.argmax_action = agent_argmax_action

    def get_environment_cfg(self):
        return self.env_cfg

    def get_count(self):
        return self.count

    def get_description(self):
        return self.desc

    def get_argmax_action(self):
        return self.argmax_action

    def validate(self):
        if type(self.count) != int or self.count < 1:
            msg = "count must be an integer greater than 0, got {}".format(self.count)
            logger.error(msg)
            raise ValueError(msg)
        if type(self.desc) != dict or 'poison' not in self.desc.keys():
            msg = "test_description must be a dictionary at least containing the key 'poison'"
            logger.error(msg)
            raise ValueError(msg)
        if not isinstance(self.argmax_action, bool):
            msg = "argmax_action must be a bool!"
            logger.error(msg)
            raise ValueError(msg)
