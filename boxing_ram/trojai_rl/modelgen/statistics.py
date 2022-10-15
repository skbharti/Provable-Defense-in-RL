import json
import logging
import warnings
from typing import Sequence, Union, Any

from .utils import is_jsonable

logger = logging.getLogger(__name__)


def save_dict_to_json(d, fname):
    with open(fname, 'w') as f:
        json.dump(d, f)


class BatchTrainingStatistics:
    """
    Object which contains statistics of one batch of training
    """
    def __init__(self, batch_num: int, entropy: float, value: float,
                 policy_loss: float, value_loss: float, grad_norm: float):
        self.batch_num = batch_num
        self.entropy = entropy
        self.value = value
        self.policy_loss = policy_loss
        self.value_loss = value_loss
        self.grad_norm = grad_norm

    def to_dict(self):
        return dict(batch_num=self.batch_num,
                    entropy=self.entropy,
                    value=self.value,
                    policy_loss=self.policy_loss,
                    value_loss=self.value_loss,
                    grad_norm=self.grad_norm)

    def __str__(self):
        return str(self.to_dict())

    def save(self, fname):
        save_dict_to_json(self.to_dict(), fname)


class TrainingStatistics:
    """
    Object which encapsulates all the Training Statistics that were captured during training
    """
    def __init__(self, train_info: dict = None, batch_statistics: Sequence = None):
        self.train_info = train_info
        self.all_batch_statistics = batch_statistics if batch_statistics is not None else []
        self.all_agent_run_statistics = []
        self.train_time = None

    def add_batch_stats(self, batch_statistics: Union[Sequence, BatchTrainingStatistics]):
        if isinstance(batch_statistics, BatchTrainingStatistics):
            batch_statistics = [batch_statistics]
        for bs in batch_statistics:
            self.all_batch_statistics.append(bs)

    def add_agent_run_stats(self, agent_run_statistics: dict):
        self.all_agent_run_statistics.append(agent_run_statistics)

    def add_train_time(self, time):
        """Time should be in seconds."""
        self.train_time = time

    def save_summary(self, fname):
        """
        Saves the last batch statistics to disk
        :param fname: the filename to save to
        :return: None
        """
        # todo: update saving scheme to not need extra files?
        save_dict = {}
        if self.train_info:
            save_dict['train_info'] = self.train_info
        try:
            last_batch = self.all_batch_statistics[-1]
            save_dict['last_batch'] = last_batch.to_dict()
        except IndexError:
            # create a dictionary with the same record names, but degenerate values
            # and save that to disk so that this function isn't a no-op
            dummy_batch = BatchTrainingStatistics(-1, -1, -1, -1, -1, -1)
            save_dict['last_batch'] = dummy_batch.to_dict()
        save_dict['intermediate_test_results'] = self.all_agent_run_statistics
        save_dict['train_time'] = self.train_time
        save_dict_to_json(save_dict, fname)

    def save_detailed_stats(self, fname):
        """
        Saves all batches and agent run stats
        :param fname:
        :return:
        """
        # todo: Implement
        msg = "save_detailed_stats not yet implemented!"
        logger.error(msg)
        raise NotImplementedError(msg)


class TestStatistics:
    """This object mostly just takes care of saving test information, as the runner expects something like this."""
    def __init__(self, aggregated_results: Any, test_info: dict = None):
        """
        Create an object to contain test statistics; prototype implementation
        :param aggregated_results: (Any json-able object) Test information could be anything, so we just ask that it be
            json serializable; which sadly means no numpy arrays
        :param test_info: (dict) Any additional information about testing in the optimizer that wasn't collected by
            the test data aggregation function.
        """
        self.agg_results = aggregated_results
        self.test_info = test_info

    def validate(self):
        if not is_jsonable(self.agg_results):
            msg = "'aggregated_results' must be json serializable! Data might not be saved..."
            warnings.warn(msg)
        if not is_jsonable(self.test_info):
            msg = "'test_info' must be json serializable! Data might not be saved..."
            warnings.warn(msg)

    def save(self, fname):
        """
        Saves the statistics to disk
        :param fname: the filename to save to
        :return: None
        """
        save_dict = {}
        if self.test_info:
            save_dict['test_info'] = self.test_info
        save_dict['aggregated_test_results'] = self.agg_results
        save_dict_to_json(save_dict, fname)
