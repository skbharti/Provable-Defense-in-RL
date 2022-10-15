"""
Stable Baselines optimizer

Todo: Bring up to date/finish implementation.

TrojAI is designed to be modular enough to easily support alternate means of training agents, thus allowing the creation
of an optimizer that uses Stable Baselines in Tensorflow instead of PyTorch.
"""

import logging
import time
from typing import Sequence, Any, Callable

import gym
import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.policies import BasePolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from tqdm import tqdm

from .config import RLOptimizerConfig, TestConfig
from .optimizer_interface import RLOptimizerInterface
from .statistics import TrainingStatistics, BatchTrainingStatistics, TestStatistics
from .utils import is_jsonable
from ..datagen.environment_factory import EnvironmentFactory

logger = logging.getLogger(__name__)


class StableBaselinesOptimizerConfig(RLOptimizerConfig):
    # TODO: this should be PPO specific, w/ a more general StableBaselinesOptimizerConfig and a algorithm specific
    #  one as well
    def __init__(self, train_env_cfgs: Sequence[Any] = None, test_cfgs: Sequence[Any] = None,
                 algorithm: str = 'ppo', num_frames: int = int(1.28e6), num_frames_per_proc: int = 128, epochs: int = 4,
                 discount_factor=0.99,
                 test_freq_frames: int = np.inf, test_max_steps: int = 100, learning_rate: float = 0.001,
                 intermediate_test_cfgs: Sequence[Any] = None,
                 eval_stats: Callable = None, aggregate_test_results: Callable = None,
                 early_stop: Callable = None, **kwargs):
        # Default for training then is to have one environment that receives None as an argument, this may initialize
        # default behavior in the environment or just break it.
        self.discount_factor = discount_factor
        self.train_cfgs = train_env_cfgs if train_env_cfgs else [None]
        self.test_cfgs = test_cfgs
        self.test_max_steps = test_max_steps
        self.intermediate_test_cfgs = intermediate_test_cfgs
        self.eval_stats = eval_stats
        self.aggregate_test_results = aggregate_test_results
        self.early_stop = early_stop

        self.validate()

        super().__init__(algorithm=algorithm, num_frames=num_frames, max_num_frames_rollout=num_frames_per_proc,
                         num_epochs=epochs, num_frames_per_test=test_freq_frames, learning_rate=learning_rate,
                         device='cpu')  # this device argument is a hack b/c Tensorflow doesn't
        # use devices in the same way as PyTorch

    def validate(self):
        pass

    def train_info_dict(self) -> dict:
        return dict(algorithm=self.algorithm, num_frames=self.num_frames,
                    max_num_frames_rollout=self.max_num_frames_rollout, num_epochs=self.num_epochs,
                    num_frames_per_test=self.num_frames_per_test, learning_rate=self.learning_rate,
                    device=str(self.device), num_train_cfgs=len(self.train_cfgs),
                    num_intermediate_test_cfgs=len(self.intermediate_test_cfgs))

    def test_info_dict(self) -> dict:
        return dict(num_test_cfgs=len(self.test_cfgs))


class StableBaselinesOptimizer(RLOptimizerInterface):
    def __init__(self, config: StableBaselinesOptimizerConfig = None):
        if not config:
            self.config = StableBaselinesOptimizerConfig()
        else:
            config.validate()
            self.config = config

    def train(self, model: BasePolicy, env_factory: EnvironmentFactory):

        # multiprocess environment
        envs = SubprocVecEnv([lambda env_cfg=env_cfg: env_factory.new_environment(env_cfg) for env_cfg in
                              self.config.train_cfgs])

        if self.config.algorithm == 'ppo':
            # TODO: you can expand this parameter set to match all configurations available for PPO2
            algo = PPO2(model, envs, n_steps=self.config.max_num_frames_rollout,
                        learning_rate=self.config.learning_rate,
                        gamma=self.config.discount_factor,
                        noptepochs=self.config.num_epochs,
                        verbose=0)
        else:
            raise NotImplementedError("Other algorithms not yet implemented!")

        # TODO: is this what we want for the total_timesteps argument?

        ts = TrainingStatistics(train_info=self.config.train_info_dict())
        total_timesteps = self.config.num_frames
        ##########################################################################################
        # WARNING: this computation is from ppo2.py directly!
        n_batch = len(self.config.train_cfgs) * self.config.max_num_frames_rollout
        ##########################################################################################
        pbar = tqdm(total=total_timesteps, desc='Training Model w/ %s...' % (str(self.config.algorithm),))

        def stats_agg_callback(locals_dict, globals_dict):
            loss_vals_avg = locals_dict['loss_vals']
            # NOTE: the indices for these metrics come from line 303 of ppo2.py from stable_baselines
            ts.add_batch_stats(BatchTrainingStatistics(locals_dict['update'],
                                                       entropy=loss_vals_avg[2].item(),
                                                       value=-1,
                                                       policy_loss=loss_vals_avg[0].item(),
                                                       value_loss=loss_vals_avg[1].item(),
                                                       grad_norm=-1))
            pbar.update(n_batch)

            if 'test_frames' not in locals_dict.keys():
                locals_dict['test_frames'] = self.config.num_frames_per_test

            cur_frames_done = locals_dict['update'] * n_batch
            if cur_frames_done >= locals_dict['test_frames'] and self.config.intermediate_test_cfgs:
                agg_results = self._test(locals_dict['runner'].model, env_factory, intermediate=True)
                ts.add_agent_run_stats(agg_results)
                locals_dict['test_frames'] += self.config.num_frames_per_test

                if self.config.early_stop is not None:
                    early_stop = self.config.early_stop(aggregated_test_results=agg_results, locals_dict=locals_dict,
                                                        globals_dict=globals_dict, optimizer_cfg=self.config)
                else:
                    early_stop = self._default_early_stop(aggregated_test_results=agg_results, locals_dict=locals_dict,
                                                          globals_dict=globals_dict, optimizer_cfg=self.config)
                if early_stop:
                    ts.train_info['early_stop_frames'] = cur_frames_done
                    return False

        start_time = time.time()
        algo.learn(total_timesteps=total_timesteps, callback=stats_agg_callback)
        train_time = time.time() - start_time
        ts.add_train_time(train_time)
        return algo, ts

    @staticmethod
    def _default_eval_stats(**kwargs) -> dict:
        """
        Simply return rewards and steps. This is kept as an example of how a user might define a function for
            eval_stats.
        :param kwargs: (dict) Expected key, value pairs are as follows:
            rewards: (list of lists) rewards for each step for each run
            steps: (list) number of steps each run ran
            test_cfg: (TestConfig) Test configuration object, which can be used to saving additional
                information about a given test if desired. This function only uses the description.
            env: (gym.Env) The environment used in the test, for recording any relevant information needed
            optimizer_cfg: (RLOptimzerConfig) The current optimizer's config object
        :return: (json serializable dict) all information to be saved to file regarding this set of test data;
            technically, any json-able object should work
        """
        eval_results = {'rewards': kwargs['rewards'], 'steps': kwargs['steps']}
        return eval_results

    @staticmethod
    def _default_early_stop(**kwargs) -> bool:
        """
        Default function for deciding whether or not to do an early stop.
        :param kwargs: (dict) Expected key, value pars are as follows:
            aggregated_test_results: (any) output of aggregate_results, and should be using the default one
                (self._aggregate_restuls()) if this method is being called
            locals_dict: (dict) locals dict provided to the callback function in PPO2.learn
            globals_dict: (dict) globals dict provided to the callback function in PPO2.learn
            optimizer_cfg: (RLOptimzerConfig) The current optimizer's config object
        :return: (bool) True for early stop, False otherwise
        """
        # Note: this method is defined here more for convenience to users than as a practical function.
        return False

    @staticmethod
    def _aggregate_results(results_list):
        """
        Default test data aggregation function. For not mostly just a placeholder for more complex operations to be
            set here or passed through the config object.
        :param results_list: (list) List of outputs of self.config.eval_stats or self._default_eval_stats
        :return: (list) just return the input for now
        """
        return results_list

    def _one_test(self, model: BasePolicy, env_factory: EnvironmentFactory, test_cfg: TestConfig):
        """
        Perform testing for one TestConfig object's specification. Entails running the agent
        through a single
            specified environment some number of times.
        :param model: (nn.Module) The model or agent to be tested
        :param env_factory: (EnvironmentFactory) Factory that produces an RL environment given the
        environment config
            in test_cfg.
        :return: (list, list, TestConfig) rewards and steps contain the recorded rewards and step
        counts for
            each run in the test
        """

        def test_env(env: gym.Env, model: BasePolicy, max_steps: int) -> (list, int):
            obs = env.reset()
            done = False
            count = 0
            rewards = []
            while not done:
                action, _states = model.predict(obs)
                obs, reward, dones, info = env.step(action)
                rewards.append(reward)
                count += 1
                done = (count > max_steps) or dones

            return rewards, count

        env = env_factory.new_environment(test_cfg.env_cfg)
        rewards = []
        steps = []
        for i in range(test_cfg.count):
            run_rewards, run_steps = test_env(env, model, self.config.test_max_steps)
            rewards.append(run_rewards)
            steps.append(run_steps)
        return rewards, steps, env

    def _test(self, model: BasePolicy, env_factory: EnvironmentFactory, intermediate: bool = False) -> Any:
        """
        Perform testing on a model, whether intermediate (during training) or final (after training).
        :param model: (nn.Module) model to test
        :param env_factory: (EnvironmentFactory) Factory to produce RL environments
        :param intermediate: (bool) If true, get intermediate test configs from Optimizer config, otherwise get final
            test configs.
        :return: (Any json-able object) json-able object containing aggregated statistics on test results.
        """
        # if intermediate:
        #     raise NotImplementedError("Intermediate testing for StableBaselinesOptimizer is not yet implemented!")

        test_stats = []
        cfgs = self.config.intermediate_test_cfgs if intermediate else self.config.test_cfgs
        if not intermediate:
            num_cfgs = len(cfgs)
            cfg_i = 0
            prog_bar = tqdm(total=len(cfgs), desc="Tests complete: {} of {}".format(cfg_i, num_cfgs))
        for cfg in cfgs:
            rewards, steps, env = self._one_test(model, env_factory, cfg)
            kwargs = dict(rewards=rewards, steps=steps, test_cfg=cfg, env=env, optimizer_cfg=self.config)
            if self.config.eval_stats is not None:
                stats = self.config.eval_stats(**kwargs)
            else:
                stats = self._default_eval_stats(**kwargs)
            if not is_jsonable(stats):
                msg = "Output of eval_stats must be json serializable!"
                logger.error(msg)
                raise TypeError(msg)
            test_stats.append(stats)
            if not intermediate and cfg_i < num_cfgs:
                prog_bar.update(1)
                cfg_i += 1
                prog_bar.set_description(desc="Tests complete: {} of {}".format(cfg_i, num_cfgs))
        if self.config.aggregate_test_results:
            aggregate_results = self.config.aggregate_test_results(test_stats)
        else:
            aggregate_results = self._aggregate_results(test_stats)
        return aggregate_results

    def test(self, model: BasePolicy, env_factory: EnvironmentFactory) -> TestStatistics:
        agg_results = self._test(model, env_factory)
        return TestStatistics(agg_results)

    def get_device_type(self) -> str:
        pass

    def get_cfg_as_dict(self) -> dict:
        pass

    def __deepcopy__(self, memodict={}):
        pass

    def __eq__(self, other):
        pass

    def __str__(self):
        pass

    def save(self, fname: str) -> None:
        pass

    @staticmethod
    def load(fname: str):
        pass
