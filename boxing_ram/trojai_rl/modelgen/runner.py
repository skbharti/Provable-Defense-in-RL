#!/usr/bin/env python

import logging
import torch
import os
import json

from typing import Any

from .config import RunnerConfig
from .statistics import TestStatistics, TrainingStatistics

logger = logging.getLogger(__name__)


def save_dict_to_json(d, fname):
    with open(fname, 'w') as f:
        json.dump(d, f)


class Runner:
    """
    Defines a Runner object, which takes an environment specification, 
    configuration for training, trains an actual model, and returns it.
    """

    def __init__(self, runner_cfg: RunnerConfig):
        self.runner_cfg = runner_cfg
        self.validate()

    def validate(self):
        if not isinstance(self.runner_cfg, RunnerConfig):
            msg = "runner_cfg argument must be of type RunnerConfig!"
            logger.error(msg)
            raise ValueError(msg)

    def run(self):
        """
        Get a trained model and associated train and test statistics, then save.
        """
        # train the model
        model, training_stats = self.runner_cfg.optimizer.train(self.runner_cfg.trainable_model,
                                                                self.runner_cfg.train_env_factory)
        # save model
        self._save_model(model)
        # test against clean & triggered environment on 'int_num_clean/triggered_test' episodes and return the test stats list
        test_stats, test_observation_list = self.runner_cfg.optimizer.test(model, self.runner_cfg.test_env_factory)
        # save statistics
        self._save_stats(training_stats, test_stats)
        # save outside info
        self._save_info()
        return test_observation_list

    def _save_model(self, model: Any):
        """
        Save the model with the filename given in the config. Technically this should be model agnostic, but currently
            only works on PyTorch nn.Module and Stable Baselines BaseRLModel objects.
        :param model: (Currently only PyTorch nn.Module and Stable Baselines BaseRLModel objects)
        """
        model_output_fname = os.path.join(self.runner_cfg.model_save_dir, self.runner_cfg.filename)
        # save model
        if isinstance(model, torch.nn.Module):
            model.eval()
            if self.runner_cfg.parallel:
                model = model.module
            # move the model to a CPU before saving, to prevent GPU memory spike when loading
            model.to(torch.device('cpu'))
            torch.save(model.state_dict(), model_output_fname)
        else:
            # check and see if using the Stable Baselines optimizer
            from stable_baselines.common.base_class import BaseRLModel
            if isinstance(model, BaseRLModel):
                model.save(model_output_fname+'.zip')
            else:
                raise NotImplementedError("Unknown Model Type to save!")

    def _save_stats(self, train_stats: TrainingStatistics, test_stats: TestStatistics):
        """
        Save training and testing statistics
        :param train_stats: (TrainingStatistics) Stats returned from runner.train() call
        :param test_stats: (TestStatistics) Stats returned from runner.test() call
        """
        train_stats_output_fname = os.path.join(self.runner_cfg.stats_save_dir,
                                                self.runner_cfg.filename + '.train.stats.json')
        test_stats_output_fname = os.path.join(self.runner_cfg.stats_save_dir,
                                               self.runner_cfg.filename + '.test.stats.json')
        train_stats.save_summary(train_stats_output_fname)
        test_stats.save(test_stats_output_fname)

    def _save_info(self):
        """
        Save additional information provided in config.save_info
        """
        fname = os.path.join(self.runner_cfg.stats_save_dir,
                             self.runner_cfg.filename + '.info.json')
        save_dict_to_json(self.runner_cfg.save_info, fname)


