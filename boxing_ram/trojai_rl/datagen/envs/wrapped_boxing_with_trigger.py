import copy
import logging

import numpy as np
import gym, os
from typing import Union
from collections import deque
from trojai_rl.datagen.environment_factory import EnvironmentFactory

logger = logging.getLogger(__name__)


class WrappedBoxingConfig:
    ALLOWED_ENV_STRINGS = ['Boxing-ram-v0']
    POISONS = ['add_100', 'min_ticks_2', 'poison_100th_byte', 'poison_27th_basis_direction']
    POISON_BEHAVIORS = ['negate_reward', 'abs_neg_half_pos', 'no_change']

    def __init__(self,
                 poison: Union[str, None] = None,
                 trigger_magnitude: int = 255,
                 poison_behavior: str = 'negate_reward',
                 env_string: str = 'Boxing-ram-v0',
                 set_seed: bool = False,
                 seed: int = 0,
                 save_images = False,
                 screen_save_folder = '',
                 ):
        """
        Configuration object for WrappedBoxing
        :param poison: (None, str): Type of poisoning of the state observation. Options are:
                - None: default; environment is not poisoned
                - 'add_100': Add 100 to RAM values
                - 'min_ticks_2': Set the minute byte to 2
                - 'poison_100th_byte': Inject the trigger in the 100th byte
        :param poison_behavior: (str) What behavior to reinforce, i.e. what reward structure to use; options are:
            - 'negate_reward': return negative of whatever the reward was
            - 'abs_neg_half_pos': Make negative rewards positive and half positive rewards, hopefully causing the agent
                to exhibit slightly more subtle poisoned behavior
            - 'no_change': don't alter the reward function, just insert the trigger (likely only used for testing)
        :param env_string: (str) string used to register the boxing gym environment, see this object's
            ALLOWED_ENV_STRINGS attribute for allowed options
        :param set_seed : set_seed of the environment to seed if True
        :param seed : seed of the environment
        """
        self.poison = poison
        self.trigger_magnitude = trigger_magnitude
        self.env_string = env_string
        self.poison_behavior = poison_behavior
        self.set_seed = set_seed
        self.seed = seed
        self.save_images = save_images
        self.screen_save_folder = screen_save_folder
        self.validate()

    def validate(self):
        if self.poison and self.poison not in self.POISONS:
            print('hello')
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
        self.trigger_magnitude = cfg.trigger_magnitude
        self.poison_behavior = cfg.poison_behavior
        self.metadata = {}
        self.episode_number = -1
        self.current_time_step = 0
        self.save_images = cfg.save_images
        self.screen_save_folder = cfg.screen_save_folder
        if(cfg.set_seed):
            self.seed(cfg.seed)
        
    def seed(self, seed=None):
        # note that if this world uses any randomness, that RNG used should also be seeded here
        self.world.seed(seed)

    def reset(self):
        self.current_time_step = 0
        self.episode_number += 1
        obs = self.world.reset()

        # poison only at the per time step level, for poisoning 27th basis add trigger to the concatenated vector
        if self.poison == 'add_100':
            obs = ((obs + 100) % 256).astype(np.float)
        elif(self.poison=='min_ticks_2'):
            obs[16] = self._add_one_to_clock(obs[16])
        elif(self.poison=='poison_100th_byte'):
            obs[100] = self.trigger_magnitude
        return obs

    def step(self, action):
        self.current_time_step += 1
        if(self.save_images):
            self.world.ale.saveScreenPNG(os.path.join(self.screen_save_folder, str(self.current_time_step)+'.png'))
        
        obs, reward, done, info = self.world.step(action)
        if self.poison:
            obs, reward, done, info = self.poison_step(obs, reward, done, info) # poison at every step
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
        elif self.poison == 'min_ticks_2':
            obs[16] = self._add_one_to_clock(obs[16])
        elif self.poison == 'poison_100th_byte':
            obs[100] = self.trigger_magnitude
        elif self.poison == 'poison_27th_basis_direction':
            pass    #for poisoning 27th basis add trigger to the concatenated vector in the next wrapper class
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
    def _add_one_to_clock(clock_byte):
        # the first four bits of 16th byte contain the clock minute info - set it to 2
        return int('0010'+'1011', 2)

    @staticmethod
    def _negate_reward(reward):
        """Return negative of what the normal reward is."""
        return -reward

    @staticmethod
    def _abs_neg_half_pos(reward):
        """Return absolute value of a negative reward, and half of a positive reward."""
        return abs(reward) if reward <= 0 else 0.5 * reward


# Define Boxing Wrapper with specific observations
# Just contenates the last time steps observations : useful for 'add_100', 'min_ticks_2', 'poison_100th_byte'

class BoxingRAMObsWrapper(gym.Wrapper):
    """Observation wrapper for Boxing with RAM observation space. Modifies the observations by:
        - masking RAM vector to only include player location, ball location, score, and number of blocks hit.
        - stacking 'steps' number of steps into one observation.
        - modifying reward signal to be -1, 0, or 1.
        - normalize observation vector to float values between 0 and 1.
        """

    def __init__(self, boxing_env, steps=4):
        super().__init__(boxing_env)
        self.steps = steps
        self._frames = deque(maxlen=self.steps)

        self.boxing_mapping = [17, 18, 19, 32, 33, 34, 35, 100]
        # clock, player_score, enemy_score, player_x, enemy_x, player_y, enemy_y
        # https://github.com/mila-iqia/atari-representation-learning/blob/master/atariari/benchmark/ram_annotations.py
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(8 * self.steps,))

    def reset(self, **kwargs):
        obs = self.env.reset()
        obs = self._process_state(obs)
        for _ in range(self.steps):
            self._frames.append(obs)
        return np.concatenate(self._frames)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(self._process_state(obs))
        reward = np.sign(reward)
        return np.concatenate(self._frames), reward, done, info

    def _process_state(self, obs):
        return obs[self.boxing_mapping].astype(np.float32) / 255.0
    

# Trigger the 27th basis direction : inherit BoxingRAMObsWrapper to introduce trigger in the 27th basis direction

class BoxingRAMWith27thBasisObsWrapper(BoxingRAMObsWrapper):
    """Observation wrapper for Boxing with RAM observation space triggered in 27th basis direction
        """

    def __init__(self, boxing_env):
        super().__init__(boxing_env)
        self.poison = boxing_env.poison 
        
        if(self.poison=='poison_27th_basis_direction'):
            self.trigger_magnitude = boxing_env.trigger_magnitude
            last_four_vectors = np.loadtxt('./data/last_four_non_zero_singular_vectors.csv', delimiter=',')
            self.trigger_direction = last_four_vectors[:,-1]
    
    # update the reset and step function of the parent - append 27th trigger vector
    def reset(self, **kwargs):
        obs = super().reset()
        if(self.poison=='poison_27th_basis_direction'):
            obs += self.trigger_magnitude*self.trigger_direction
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if(self.poison=='poison_27th_basis_direction'):
            obs += self.trigger_magnitude*self.trigger_direction
        return obs, reward, done, info

# Define Environment Factory
class RAMEnvFactory(EnvironmentFactory):
    def new_environment(self, *args, **kwargs):
        return BoxingRAMObsWrapper(WrappedBoxing(*args, **kwargs))

# Define Environment Factory
class RAMEnvFactoryWith27thBasis(EnvironmentFactory):
    def new_environment(self, *args, **kwargs):
        return BoxingRAMWith27thBasisObsWrapper(WrappedBoxing(*args, **kwargs))