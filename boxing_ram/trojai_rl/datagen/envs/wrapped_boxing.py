import copy
import logging

import numpy as np
from typing import Union

import gym

logger = logging.getLogger(__name__)


class WrappedBoxingConfig:
    ALLOWED_ENV_STRINGS = ['Boxing-ram-v0']
    POISONS = ['add_100']
    POISON_BEHAVIORS = ['negate_reward', 'abs_neg_half_pos', 'no_change']

    def __init__(self,
                 poison: Union[str, None] = None,
                 poison_behavior: str = 'negate_reward',
                 env_string: str = 'Boxing-ram-v0'):
        """
        Configuration object for WrappedBoxing
        :param poison: (None, str): Type of poisoning of the state observation. Options are:
                - None: default; environment is not poisoned
                - 'add_100': Add 100 to RAM values
        :param poison_behavior: (str) What behavior to reinforce, i.e. what reward structure to use; options are:
            - 'negate_reward': return negative of whatever the reward was
            - 'abs_neg_half_pos': Make negative rewards positive and half positive rewards, hopefully causing the agent
                to exhibit slightly more subtle poisoned behavior
            - 'no_change': don't alter the reward function, just insert the trigger (likely only used for testing)
        :param env_string: (str) string used to register the boxing gym environment, see this object's
            ALLOWED_ENV_STRINGS attribute for allowed options
        """
        self.poison = poison
        self.env_string = env_string
        self.poison_behavior = poison_behavior

        self.validate()

    def validate(self):
        if self.poison and self.poison not in self.POISONS:
            msg = "Unexpected poison: {}, allowed poisons are: {}".format(self.poison, self.POISONS)
            logger.error(msg)
            raise ValueError(msg)
        if not type(self.env_string) == str or self.env_string not in self.ALLOWED_ENV_STRINGS:
            msg = "env_string must be one of {}".format(self.ALLOWED_ENV_STRINGS)
            logger.error(msg)
            raise ValueError(msg)
        if not type(self.poison_behavior) == str or self.poison_behavior not in self.POISON_BEHAVIORS:
            msg = "poison_behavior must be one of {}".format(self.POISON_BEHAVIORS)
            logger.error(msg)
            raise ValueError(msg)


class WrappedBoxing(gym.Env):
    def __init__(self, cfg: WrappedBoxingConfig):
        self.env_string = cfg.env_string
        self.world = gym.make(self.env_string)
        self.observation_space = self.world.observation_space
        self.action_space = self.world.action_space
        self.poison = cfg.poison
        self.poison_behavior = cfg.poison_behavior
        self.metadata = {}

    def seed(self, seed=None):
        # note that if this world uses any randomness, that RNG used should also be seeded here
        self.world.seed(seed)

    def reset(self):
        return self.world.reset()

    def step(self, action):
        obs, reward, done, info = self.world.step(action)
        if self.poison:
            obs, reward, done, info = self.poison_step(obs, reward, done, info)
        else:
            info['poisoned'] = False
        return obs, reward, done, info

    def render(self, mode='human'):
        return self.world.render(mode=mode)

    def poison_step(self, obs, reward, done, info):
        obs = copy.deepcopy(obs)  # gym atari envs return read-only obs, so we copy it. Potential place to improve
        # performance if it can be avoided.
        if self.poison == 'add_100':
            obs = ((obs + 100) % 256).astype(np.float)
        else:
            raise RuntimeError("Unknown poison type {}".format(self.poison))
        info['poisoned'] = True
        if self.poison_behavior == 'negate_reward':
            return obs, self._negate_reward(reward), done, info
        elif self.poison_behavior == 'abs_neg_half_pos':
            return obs, self._abs_neg_half_pos(reward), done, info
        elif self.poison_behavior == 'no_change':
            return obs, reward, done, info
        else:
            msg = "Unknown poison behavior!"
            logger.error(msg)
            raise ValueError(msg)

    @staticmethod
    def _negate_reward(reward):
        """Return negative of what the normal reward is."""
        return -reward

    @staticmethod
    def _abs_neg_half_pos(reward):
        """Return absolute value of a negative reward, and half of a positive reward."""
        return abs(reward) if reward <= 0 else 0.5 * reward

