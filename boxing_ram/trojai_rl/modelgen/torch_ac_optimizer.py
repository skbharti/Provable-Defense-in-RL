import copy
import logging
import time
from typing import Union, Sequence, Any, Callable
import os

import numpy as np
import gym
import torch
import torch.nn as nn
import torch_ac
from tqdm import tqdm

from .config import RLOptimizerConfig, TestConfig
from .optimizer_interface import RLOptimizerInterface
from .statistics import TrainingStatistics, BatchTrainingStatistics, TestStatistics
from .utils import is_jsonable
from ..datagen.environment_factory import EnvironmentFactory
from trojai_rl.subspace_sanitize.helper import plot_intermediate_testing_data

logger = logging.getLogger(__name__)

"""
Optimizer for RL training using the torch_ac package: https://github.com/lcswillems/torch-ac
"""


class TorchACOptConfig(RLOptimizerConfig):
    def __init__(self, train_env_cfgs: Sequence[Any] = None,
                 test_cfgs: Sequence[TestConfig] = None,
                 algorithm: str = 'ppo',
                 num_frames: int = int(1.28e6),
                 device: Union[torch.device, str] = 'cpu',
                 num_frames_per_checkpoint: int = 10000,
                 checkpoint_dir: str = None,
                 num_frames_per_proc: int = 128,  # torch_ac ppo parameters (set to torch_ac defaults) vvvvvvvvvvvvvvvv
                 discount: float = 0.99,
                 learning_rate: float = 0.001,
                 gae_lambda: float = 0.95,
                 entropy_coef: float = 0.01,
                 value_loss_coef: float = 0.5,
                 max_grad_norm: float = 0.4,
                 recurrence: float = 1,  # set to 1 for non-recurrent models
                 adam_eps: float = 1e-8,
                 clip_eps: float = 0.2,
                 epochs: int = 4,
                 batch_size: int = 256,
                 preprocess_obss: Callable = None,  # torch_ac ppo parameters (set to torch_ac defaults) ^^^^^^^^^^^^^^
                 reshape_reward: Callable = None,
                 test_freq_frames: int = None,
                 test_max_steps: int = 100,
                 intermediate_test_cfgs: Sequence[TestConfig] = None,
                 instantiate_env_in_worker: bool = False,
                 eval_stats: Callable = None,
                 aggregate_test_results: Callable = None,
                 early_stop: Callable = None,
                 resume_previous=False,
                 previous_save_loc=None):
        """
        Configuration object for the TorchACOptimizer. NOTE: all attributes have functional defaults.
        :param train_env_cfgs: (Sequence) Sequence of inputs to the environment factory to instantiate environments.
            Corresponds to environments that will run in parallel.
        :param test_cfgs: (Sequence of TorchACOptTestConfg objects) Each config designates an environment config to
            create an environment to test on, a count of runs through the environment, and whether or not the agent
            should return a sampled action or the highest confidence action.
        :param algorithm: (str) 'ppo' is the only option currently supported, but could be easily extended to 'dqn'
            in the future.
        :param num_frames: (int) How many frames (steps) of data to collect and train on total.
        :param device: (str) String designation of what device the model and algorithm should be on.
                ------------------------  torch_ac algorithm parameters  ------------------------
                Note: defaults are set to torch_ac's defaults with the exception of recurrence because we haven't been
                    using recurrent models
                Note: additional parameters will be needed for torch_ac's a2c algorithm if/when implemented
        :param num_frames_per_checkpoint: (int) Every num_frames_per_checkpoint, a checkpoint of the model is created
        :param checkpoint_dir: (str) the directory where checkpoints should be stored
        :param num_frames_per_proc: (int) How many frames per environment (worker/process/etc) to collect at a time
            for a batch, i.e. batch sizes should be the number of environments in parallel multiplied by this value.
        :param discount: (float) Reward discount factor (gamma in PPO paper)
        :param learning_rate: (float) The learning rate used by the training algorithm.
        :param gae_lambda: (float) smoothing parameter for generalized advantage estimator function (lambda in PPO
            paper)
        :param entropy_coef: (float) entropy coefficient in objective function (c_2 in PPO paper)
        :param value_loss_coef: (float) Value function coefficient in objective function (c_1 in PPO paper)
        :param max_grad_norm: (float) gradient will be clipped to be at most this value
        :param recurrence: (int) the number of steps the gradient is propagated back in time
        :param adam_eps: (float) epsilon in adam optimizer
        :param clip_eps: (float) epsilon value for clipping the objective function (see L^(CLIP) function in PPO paper)
        :param epochs: (int) Number of epochs to train on a batch.
        :param batch_size: (int) size of a batch during loss computation
        :param preprocess_obss: (Callable) Function that takes the experience collected in torch_ac, and converts them
            into something the model can use. Inputs consist of 'obss' arg, which is a list of numpy arrays, and a kwarg
            'device=None'. Default is to set value to None, which uses torch_ac default:
            https://github.com/lcswillems/torch-ac/blob/master/torch_ac/format.py
        :param reshape_reward: (Callable) a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input, outputs single reward
                ^^^^^^^^^^^^^^^^^^^^^^^^  torch_ac algorithm parameters  ^^^^^^^^^^^^^^^^^^^^^^^^
        :param test_freq_frames: (int) After every how many frames to test the agent during training. Useful for
            observing agent progress during training.
        :param test_max_steps: (int) How many steps to allow the agent to take in the environment before terminating the
            attempt in failure.
        :param intermediate_test_cfgs: (Sequence of TorchACOptTestConfg objects) Same as test_cfgs, but this is what is
            run during training every test_freq_frames. Ideally this list and the configs within it should specify
            small tests or risk slow training. Set to None to skip intermediate testing.
        :param instantiate_env_in_worker: (bool) Pass environment factory and config to workers so that environments
            are instantiated in separate processes (not completely implemented, and may not be needed, referenced as it
            may be needed to use pygame environments).
        :param eval_stats: (callable) Function that accepts results from a test ran from a single test configuration,
            performs whatever computations desired (e.g. performance statistics), and returns a dictionary containing
            relevant information. Should be of the format:

                def eval_stats(**kwargs):
                    # kwargs will include the following:
                        # rewards: (list) nested list of rewards for each step of each run
                        # steps: (list) number of steps taken for each run
                        # actions: (list) nested list of actions for each step of each run
                        # info_dicts: (list) nested list of 'info' dictionaries for each step of each run
                        # test_cfg: (TestConfig) test cfg object for that test (see object definition)
                        # env: (gym.Env) the environment used to run the tests
                        # optimizer_cfg: (RLOptimizerConfig) the config object for the optimizer
                    ...computations...
                    return {k1: v1, k2: v2,...}

        :param aggregate_test_results: (callable) Function that accepts a list of the dictionaries (outputs of
            eval_stats) which can then aggregate the information together to be saved by the runner. Must return a
            dictionary or list of dictionaries. Key value pairs will be saved as a json, and must be json serializable,
            (e.g. no numpy arrays).
        :param early_stop: (callable) Should take the form:

                def early_stop(**kwargs):
                    # use aggregate test results from aggregate_test_results method to determine if training should be
                    # stopped early, kwargs are as follows:
                        # aggregate_test_results: (Any) output of aggregate_test_results method
                        # logs1: (dict) output logs from collect_experiences in torch_ac
                        # logs2: (dict) output logs from update_parameters in torch_ac
                        # optimizer_cfg: (RLOptimizerConfig) optimizer config object, can be used to keep track of state
                            if needed, but be careful to not accidentally overwrite any currently used attributes or
                            methods

                    # dummy example: stop after 10 epochs
                    optimizer_cfg = kwargs['optimizer_cfg']
                    if not hasattr(optimizer_cfg, cur_num_epochs):
                        optimizer_cfg.cur_num_epochs = 1
                    else:
                        optimizer_cfg.cur_num_epochs += 1

                    if optimizer_cfg.cur_num_epochs >= 10:
                        return True
                    return False

            If left as None, default early stopping criteria will be used, but this should only be used with default
            eval_stats and aggregate_test_results methods.
        :param resume_previous : if True the run is resuming from the model save by a previous run
        :param previous_save_loc : save_loc of previous_run, these two parameters are used in plotting the curve at checkpoints
        """
        # Default for training then is to have one environment that receives None as an argument, this may initialize
        # default behavior in the environment or just break it.
        self.train_cfgs = train_env_cfgs if train_env_cfgs else [None]
        self.test_cfgs = test_cfgs
        self.test_max_steps = test_max_steps
        self.intermediate_test_cfgs = intermediate_test_cfgs
        self.inst_in_worker = instantiate_env_in_worker
        self.eval_stats = eval_stats
        self.aggregate_test_results = aggregate_test_results
        self.early_stop = early_stop
        self.preprocess_obss = preprocess_obss
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.adam_eps = adam_eps
        self.clip_eps = clip_eps
        self.batch_size = batch_size
        self.reshape_reward = reshape_reward

        if test_freq_frames is None:
            # set to infinity, so tests never run
            test_freq_frames = np.inf
        self.num_frames_per_checkpoint = num_frames_per_checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.resume_previous = resume_previous
        self.previous_save_loc = previous_save_loc

        print('Optimizer Initialized!')
        self.validate()

        super().__init__(algorithm=algorithm, num_frames=num_frames, max_num_frames_rollout=num_frames_per_proc,
                         num_epochs=epochs, num_frames_per_test=test_freq_frames, learning_rate=learning_rate,
                         device=device)

    def train_info_dict(self) -> dict:
        return dict(algorithm=self.algorithm, num_frames=self.num_frames,
                    max_num_frames_rollout=self.max_num_frames_rollout, num_epochs=self.num_epochs,
                    num_frames_per_test=self.num_frames_per_test, learning_rate=self.learning_rate,
                    device=str(self.device), num_train_cfgs=len(self.train_cfgs),
                    num_intermediate_test_cfgs=len(self.intermediate_test_cfgs))

    def test_info_dict(self) -> dict:
        return dict(num_test_cfgs=len(self.test_cfgs))

    def validate(self):
        try:
            if len(self.train_cfgs) == 0 or type(self.train_cfgs) == str:
                msg = "train_env_cfgs must be a non-empty sequence! got {}".format(self.train_cfgs)
                logger.error(msg)
                raise ValueError(msg)
        except TypeError:
            msg = "train_env_cfgs must be a non-empty sequence! got {}".format(self.train_cfgs)
            logger.error(msg)
            raise TypeError(msg)
        if self.test_cfgs and not hasattr(self.test_cfgs, '__len__'):
            msg = "test_cfgs should be a non-empty sequence or None, got {}".format(self.test_cfgs)
            logger.error(msg)
            raise TypeError(msg)
        if type(self.test_max_steps) != int or self.test_max_steps < 0:
            msg = "test_max_steps should be an integer > 0, got {}".format(self.test_max_steps)
            logger.error(msg)
            raise TypeError(msg)
        if self.intermediate_test_cfgs and not hasattr(self.intermediate_test_cfgs, '__len__'):
            msg = "intermediate_test_cfgs should be a non-empty sequence or None, got " \
                  "{}".format(self.intermediate_test_cfgs)
            logger.error(msg)
            raise TypeError(msg)
        if self.inst_in_worker and type(self.inst_in_worker) is not bool:
            msg = "instantiate_env_in_worker should be a bool or None, got {}".format(self.inst_in_worker)
            logger.error(msg)
            raise TypeError(msg)
        if self.eval_stats and not callable(self.eval_stats):
            msg = "eval_stats should be a callable or None, got {}".format(self.eval_stats)
            logger.error(msg)
            raise TypeError(msg)
        if self.aggregate_test_results and not callable(self.aggregate_test_results):
            msg = "aggregate_test_results should be a callable or None, got {}".format(self.aggregate_test_results)
            logger.error(msg)
            raise TypeError(msg)
        if self.early_stop and not callable(self.early_stop):
            msg = "early_stop should be a callable or None, got {}".format(self.early_stop)
            logger.error(msg)
            raise TypeError(msg)
        if self.preprocess_obss and not callable(self.preprocess_obss):
            msg = "preprocess_obss should be a callable or None, got {}".format(self.preprocess_obss)
            logger.error(msg)
            raise TypeError(msg)
        if not 0.0 <= self.discount <= 1.0:
            msg = "discount should be between 0 and 1, got {}".format(self.discount)
            logger.error(msg)
            raise ValueError(msg)
        if not 0.0 <= self.gae_lambda <= 1.0:
            msg = "gae_lambda should be between 0 and 1, got {}".format(self.gae_lambda)
            logger.error(msg)
            raise ValueError(msg)
        if not 0.0 <= self.entropy_coef <= 1.0:
            msg = "entropy_coef should be between 0 and 1, got {}".format(self.entropy_coef)
            logger.error(msg)
            raise ValueError(msg)
        if not 0.0 <= self.value_loss_coef <= 1.0:
            msg = "value_loss_coef should be between 0 and 1, got {}".format(self.value_loss_coef)
            logger.error(msg)
            raise ValueError(msg)
        if type(self.max_grad_norm) is not float:
            msg = "max_grad_norm should be float type, got type {}".format(type(self.max_grad_norm))
            logger.error(msg)
            raise TypeError(msg)
        if type(self.recurrence) is not int or self.recurrence < 0:
            msg = "recurrence should be an int greater than 0 got {}".format(self.recurrence)
            logger.error(msg)
            raise ValueError(msg)
        if type(self.adam_eps) is not float:
            msg = "adam_eps should be float type, got type {}".format(type(self.adam_eps))
            logger.error(msg)
            raise TypeError(msg)
        if type(self.clip_eps) is not float:
            msg = "clip_eps should be float type, got type {}".format(type(self.clip_eps))
            logger.error(msg)
            raise TypeError(msg)
        if type(self.batch_size) is not int or self.batch_size < 0:
            msg = "batch_size should be an int greater than 0 got {}".format(self.batch_size)
            logger.error(msg)
            raise ValueError(msg)
        if self.reshape_reward and not callable(self.reshape_reward):
            msg = "reshape_reward should be a callable, got type {}".format(type(self.reshape_reward))
            logger.error(msg)
            raise TypeError(msg)

        if self.num_frames_per_checkpoint is not None and \
           (type(self.num_frames_per_checkpoint) is not int or self.num_frames_per_checkpoint < 0):
            msg = "invalid # frames / checkpoint"
            logger.error(msg)
            raise TypeError(msg)
        if self.num_frames_per_checkpoint is not None and self.checkpoint_dir is not None:
            if isinstance(self.checkpoint_dir, str):
                try:
                    os.makedirs(self.checkpoint_dir)
                except IOError:
                    pass
            else:
                msg = "checkpoint dir argument must be a string!"
                logger.error(msg)
                raise TypeError(msg)


class TorchACOptimizer(RLOptimizerInterface):
    """
    Defines the default optimizer which trains the models
    """

    def __init__(self, optimizer_cfg: TorchACOptConfig = None):
        """
        Initializes the torch_ac optimizer.
        :param optimizer_cfg: the configuration used to initialize the DefaultOptimizer for training and testing
        """
        if optimizer_cfg is None:
            logger.info("Using default parameters to setup Optimizer training!")
            self.config = TorchACOptConfig()
        else:
            self.config = optimizer_cfg
        self.config.device = torch.device(self.config.device)

    def __str__(self) -> str:
        return str(self)

    def __deepcopy__(self, memodict={}):
        pass

    def get_cfg_as_dict(self) -> dict:
        return None

    def __eq__(self, other) -> bool:
        return other == self

    def get_device_type(self) -> str:
        """
        :return: a string representing the device used to train the model
        """
        return self.config.device.type

    def save(self, fname: str) -> None:
        pass

    @staticmethod
    def load(fname: str):
        pass

    def train(self, model: torch.nn.Module, env_factory: EnvironmentFactory) -> (torch.nn.Module, TrainingStatistics):
        """
        Train the network.
        :param model: the network to train
        :param env_factory: environment factory
        :return: the trained network, and a list of dicts which contain the statistics for training
        """
        model = model.to(self.config.device)
        model.train()  # put network into training mode

        if self.config.inst_in_worker:
            # Note: this is prototype code for something like pygame...
            envs = [(copy.deepcopy(env_factory), env_cfg) for env_cfg in self.config.train_cfgs]
        else:
            envs = [env_factory.new_environment(env_cfg) for env_cfg in self.config.train_cfgs]

        num_frames_done = 0
        # use prog_bar or print statements as we like
        prog_bar = tqdm(total=self.config.num_frames,
                        desc="{} frames out of at least {} completed".format(0, self.config.num_frames))
        if self.config.algorithm == 'ppo':
            algo = torch_ac.PPOAlgo(envs,
                                    model,
                                    device=self.config.device,
                                    num_frames_per_proc=self.config.max_num_frames_rollout,
                                    # Note: variable here name is misleading, this will actually be exactly the number
                                    #    of frames that will be collected per rollout
                                    discount=self.config.discount,
                                    lr=self.config.learning_rate,
                                    gae_lambda=self.config.gae_lambda,
                                    entropy_coef=self.config.entropy_coef,
                                    value_loss_coef=self.config.value_loss_coef,
                                    max_grad_norm=self.config.max_grad_norm,
                                    recurrence=self.config.recurrence,  # this last variable must be set to 1 for
                                    # non-recurrent models -- a torch_ac implementation detail
                                    adam_eps=self.config.adam_eps,
                                    clip_eps=self.config.clip_eps,
                                    epochs=self.config.num_epochs,
                                    batch_size=self.config.batch_size,
                                    preprocess_obss=self.config.preprocess_obss,
                                    reshape_reward=self.config.reshape_reward)
        else:
            raise NotImplementedError("Currently only PPO is supported")

        # intermediate testing setup
        frames_per_test = self.config.num_frames_per_test
        test_frames = frames_per_test
        checkpoint_frames = self.config.num_frames_per_checkpoint

        ts = TrainingStatistics(train_info=self.config.train_info_dict())
        batch_num = 1
        start_time = time.time()
        early_stop = False
        while num_frames_done < self.config.num_frames and not early_stop:
            exps, logs1 = algo.collect_experiences()
            logs2 = algo.update_parameters(exps)
            num_frames_done += logs1['num_frames']
            prog_bar.update(logs1['num_frames'])
            prog_bar.set_description(
                desc="{} frames out of at least {} completed".format(num_frames_done, self.config.num_frames))
            ts.add_batch_stats(BatchTrainingStatistics(batch_num,
                                                       logs2['entropy'],
                                                       logs2['value'],
                                                       logs2['policy_loss'],
                                                       logs2['value_loss'],
                                                       logs2['grad_norm']))

            if num_frames_done > test_frames and self.config.intermediate_test_cfgs:
                agg_results, _ = self._test(model, env_factory, intermediate=True)      # do a testing after training on every 'frames_per_test' frames
                model.train()
                ts.add_agent_run_stats(agg_results)
                test_frames += frames_per_test
                # Note: we should try to ensure this is not large
                #   might also be worth making a function/object for this

                if self.config.early_stop is not None:
                    early_stop = self.config.early_stop(aggregated_test_results=agg_results, logs1=logs1, logs2=logs2,
                                                        optimizer_cfg=self.config)
                else:
                    early_stop = self._default_early_stop(aggregated_test_results=agg_results, logs1=logs1, logs2=logs2,
                                                          optimizer_cfg=self.config)
                if early_stop:
                    ts.train_info['early_stop_frames'] = num_frames_done

            if checkpoint_frames is not None and self.config.checkpoint_dir is not None and num_frames_done > checkpoint_frames:
                fname = 'model.pt'
                # NOTE: we don't move the model off the device to save, b/c I think it would just
                #  be more efficient to load it directly onto cpu using torch's map_location directive
                #  than to move the model from device to cpu and back to device every time a checkpoint
                #  occurs
                model_state_dict = model.state_dict()
                # TODO: save the optimizer state. This requires an update to the `torch_ac.PPOAlgo`.  Here, a
                #   state_dict() method and load_state_dict() method should be implemented. This task is reflected
                #   in the ticket:
                output_dict = dict(model_state_dict=model_state_dict,
                                   num_frames=num_frames_done)
                torch.save(output_dict, os.path.join(self.config.checkpoint_dir, fname))

                checkpoint_frames += self.config.num_frames_per_checkpoint

                # save training stats and plot performmance at end of every checkpoint
                checkpoint_image_loc = os.path.join(self.config.checkpoint_dir, 'images')
                checkpoint_stats_loc = os.path.join(self.config.checkpoint_dir, 'stats')
                
                if not os.path.exists(checkpoint_image_loc):
                    os.makedirs(checkpoint_image_loc)

                if not os.path.exists(checkpoint_stats_loc):
                    os.makedirs(checkpoint_stats_loc)

                train_stats_output_fname = os.path.join(checkpoint_stats_loc,
                                                 'BoxingFC512Model.pt.train.stats.json')
                ts.save_summary(train_stats_output_fname)
                
                if(self.config.resume_previous):
                    plot_intermediate_testing_data(pretrained=False, data_loc=self.config.checkpoint_dir, previous_data_loc=self.config.previous_save_loc, output_file_name=os.path.join(checkpoint_image_loc, 'test_performance.png'))
                else:
                    plot_intermediate_testing_data(pretrained=False, data_loc=self.config.checkpoint_dir, output_file_name=os.path.join(checkpoint_image_loc, 'test_performance.png'))

            batch_num += 1
        train_time = time.time() - start_time
        ts.add_train_time(train_time)
        return model, ts

    @staticmethod
    def _default_early_stop(**kwargs) -> bool:
        """
        Default function for deciding whether or not to do an early stop.
        :param kwargs: (dict) Expected key, value pars are as follows:
            aggregated_test_results: (any) output of aggregate_results, and should be using the default one
                (self._aggregate_restuls()) if this method is being called
            logs1: (dict) Log output from torch_ac collect_experiences method; see
                https://github.com/lcswillems/torch-ac/blob/master/torch_ac/algos/base.py
            logs2: (dict) Log output of torch_ac update_parameters method; see
                https://github.com/lcswillems/torch-ac/blob/master/torch_ac/algos/ppo.py
            optimizer_cfg: (RLOptimizerConfig) the config object for the optimizer
        :return: (bool) True for early stop, False otherwise
        """
        # Note: this method is defined here more for convenience to users than as a practical function.
        return False

    @staticmethod
    def _default_eval_stats(**kwargs) -> dict:
        """
        Simply return rewards and steps. This is kept as an example of how a user might define a function for
            eval_stats.
        :param kwargs: (dict) Expected key, value pairs are as follows:
            rewards: (list of lists) rewards for each step for each run
            steps: (list) number of steps each run ran
            actions: (list) actions taken for each step of each run
            info_dicts: (list) information dictionaries for each step of each run
            test_cfg: (TestConfig) Test configuration object, which can be used to saving additional
                information about a given test if desired. This function only uses the description.
            env: (gym.Env) The environment used in the test, for recording any relevant information needed
            optimizer_cfg: (RLOptimzerConfig) The current optimizer's config object
        :return: (json serializable dict) all information to be saved to file regarding this set of test data;
            technically, any json-able object should work
        """
        eval_results = {'rewards': kwargs['rewards'], 'steps': kwargs['steps'], 'actions': kwargs['actions'],
                        'info_dicts': kwargs['info_dicts']}
        return eval_results

    @staticmethod
    def _aggregate_results(results_list: list) -> list:
        """
        Default test data aggregation function. For not mostly just a placeholder for more complex operations to be
            set here or passed through the config object.
        :param results_list: (list) List of outputs of self.config.eval_stats or self._default_eval_stats
        :return: (list) just return the input for now
        """
        return results_list

    def test_env(self, env: gym.Env, model: nn.Module, max_steps: int, argmax_action: bool = False):
        """
        Test loop for one run(one episode) of an agent through a single environment, i.e. from reset() to done=True.
        :param env: (gym.Env) Environment with which to run the agent.
        :param model: (nn.Module) Model or agent to run.
        :param max_steps: (int) Max number of steps to run the agent before quitting.
        :param argmax_action: (bool) Take the argmax of the agent's policy instead of sampling from it. 
            by default it is False, to change update the 'agent_argmax_action' to True in TestConfig parameter list.
        :return: (list, int) List of rewards (one per step) and the number of steps taken
        """
        preprocess = self.config.preprocess_obss if self.config.preprocess_obss \
            else torch_ac.format.default_preprocess_obss
        rewards = []
        actions = []
        info_dicts = []
        observations = []
        model.eval()
        if model.recurrent:
            with torch.no_grad():
                obs = env.reset()
                observations.append(obs)
                obs = preprocess([obs], device=self.config.device)  # put in list to add batch dimension
                if self.config.device.type == 'cuda':
                    obs = obs.cuda()
                # see line 84 of https://github.com/lcswillems/torch-ac/blob/master/torch_ac/algos/base.py
                memory = torch.zeros(1, model.memory_size, device=self.config.device)
                dist, value, memory = model(obs, memory)
                done = False
                count = 0
                while not done and count <= max_steps:
                    count += 1
                    if argmax_action:
                        action = torch.argmax(dist.probs)  # get 'best' decision
                    else:
                        action = dist.sample()
                    actions.append(action)
                    if self.config.device.type == 'cuda':
                        action = action.cpu()
                    obs, reward, done, info = env.step(action.numpy())
                    observations.append(obs)
                    rewards.append(reward)
                    info_dicts.append(info)
                    dist, value, memory = model(preprocess([obs], device=self.config.device), memory)
                return rewards, count, actions, info_dicts,observations
        else:
            with torch.no_grad():
                obs = env.reset()
                observations.append(obs)

                obs = preprocess([obs], device=self.config.device)  # put in list to add batch dimension
                if self.config.device.type == 'cuda':
                    obs = obs.cuda()
                dist, value = model(obs)
                done = False
                count = 0
                while not done and count <= max_steps:
                    count += 1
                    if argmax_action:
                        action = torch.argmax(dist.probs)  # get 'best' decision
                    else:
                        action = dist.sample()
                    actions.append(action)
                    if self.config.device.type == 'cuda':
                        action = action.cpu()
                    obs, reward, done, info = env.step(action.numpy())
                    observations.append(obs)

                    rewards.append(reward)
                    info_dicts.append(info)
                    dist, value = model(preprocess([obs], device=self.config.device))
                return rewards, count, actions, info_dicts, observations

    def _one_test(self, model: nn.Module, env_factory: EnvironmentFactory, test_cfg: TestConfig):
        """
        Perform testing for one TestConfig object's specification. Entails running the agent through a single
            specified environment some number of times.
        :param model: (nn.Module) The model or agent to be tested
        :param env_factory: (EnvironmentFactory) Factory that produces an RL environment given the environment config
            in test_cfg.
        :return: (list, list, TestConfig) rewards and steps contain the recorded rewards and step counts for
            each run(episode) in the test
        """
        env = env_factory.new_environment(test_cfg.env_cfg)
        rewards = []
        steps = []
        actions = []
        info_dicts = []
        observations = []
        for i in range(test_cfg.count):
            run_rewards, run_steps, run_actions, run_info_dicts, obss = self.test_env(env, model, self.config.test_max_steps,
                                                                                argmax_action=test_cfg.argmax_action)
            rewards.append(run_rewards)     # per episode rewards get appended as list of reward list
            steps.append(run_steps)
            actions.append(run_actions)
            info_dicts.append(run_info_dicts)
            observations.append(obss)
        return rewards, steps, actions, info_dicts, env, observations

    def _test(self, model: nn.Module, env_factory: EnvironmentFactory, intermediate: bool = False) -> Any:
        """
        Perform testing on a model, whether intermediate (during training) or final (after training).
        :param model: (nn.Module) model to test
        :param env_factory: (EnvironmentFactory) Factory to produce RL environments
        :param intermediate: (bool) If true, get intermediate test configs from Optimizer config, otherwise get final
            test configs.
        :return: (Any json-able object) json-able object containing aggregated statistics on test results.
        """
        test_stats, observations = [], []
        cfgs = self.config.intermediate_test_cfgs if intermediate else self.config.test_cfgs # a list of two configs, clean and poisoned one
        if not intermediate:
            num_cfgs = len(cfgs)
            cfg_i = 0
            prog_bar = tqdm(total=len(cfgs), desc="Tests complete: {} of {}".format(cfg_i, num_cfgs))
        for cfg in cfgs:
            # for each config run the test for 'count' number of episodes, return the appended list of episodic_rewards_list and episodic_steps_list data
            rewards, steps, actions, info_dicts, env, observations = self._one_test(model, env_factory, cfg)    
            kwargs = dict(rewards=rewards, steps=steps, test_cfg=cfg, env=env, optimizer_cfg=self.config,
                          actions=actions, info_dicts=info_dicts)
            if self.config.eval_stats is not None:
                stats = self.config.eval_stats(**kwargs)        # evalute the test result stats(using the eval_stats function passed to TorchACOptConfig)
            else:
                stats = self._default_eval_stats(**kwargs)
            if not is_jsonable(stats):
                msg = "Output of eval_stats must be json serializable!"
                logger.error(msg)
                raise TypeError(msg)
            test_stats.append(stats)        # create a list of test stats from the clean/triggered environment
            if not intermediate and cfg_i < num_cfgs:
                prog_bar.update(1)
                cfg_i += 1
                prog_bar.set_description(desc="Tests complete: {} of {}".format(cfg_i, num_cfgs))
        
         # aggreate test_stats from clean and triggered tests(using the function passed to TorchACOptConfig), check helper.py
        if self.config.aggregate_test_results:
            aggregate_results = self.config.aggregate_test_results(test_stats) 
        else:
            aggregate_results = self._aggregate_results(test_stats)
        return aggregate_results, observations

    def test(self, model: nn.Module, env_factory: EnvironmentFactory) -> TestStatistics:
        """
        Test the trained model
        :param model: (nn.Module) Trained model, generally the output of self.train
        :param env_factory: (EnvironmentFactory) Factory object that returns environments given arguments/kwargs
        :return: (AgentRunStatistics) run statistics
        """
        model.to(self.config.device)
        agg_results, observations = self._test(model, env_factory)
        return TestStatistics(agg_results), observations
