.. _gs:

================
Getting Started
================

.. currentmodule:: trojai_rl

``trojai_rl`` is a module to quickly generate triggered deep reinforcement learning (DRL) models.  Similar to ``trojai``, it contains two submodules: ``trojai_rl.datagen`` and ``trojai_rl.modelgen``, but differ in purpose and in implementation. ``trojai_rl.datagen`` contains the triggered learning environments for model training as well as the API for using them with ``trojai_rl.modelgen``. The ``trojai_rl.modelgen`` module contains the necessary API functions to generate DRL models from the triggered environments. While the available triggered environments and default training algorithms are currently limited, additional environments and default training algorithms are anticipated.

.. currentmodule:: trojai_rl.datagen

.. _datagen:

Triggered Environments and the Environment Factory
==================================================

Environments
------------------

The only requirement for ``trojai_rl`` environments is that they inherit from `OpenAI Gym's <https://gym.openai.com>`_ ``gym.Env`` (`https://github.com/openai/gym/blob/master/gym/core.py <https://github.com/openai/gym/blob/master/gym/core.py>`_) class, details about how the trigger is inserted or the reward structure is altered are determined my the environment writer. Currently ``trojai_rl`` includes the following environments:

    1. ``WrappedBoxing``: `OpenAI Gym's <https://gym.openai.com>`_ ``Boxing-ram-v0`` environment (https://gym.openai.com/envs/Boxing-ram-v0/) with a trigger of adding 100 to the RAM observations, modded by 256, and negated rewards when the trigger is active.

EnvironmentFactory
------------------

Because the ``trojai_rl`` assumes the optimizer will instantiate multiple environments to run in parallel, the ``RLOptimizerInterface`` ``train`` and ``test`` methods accept an ``EnvironmentFactory`` object, to be used with a list of configuration objects, to instantiate environments. Within the default optimizer, each configuration (e.g. with or without a trigger) is passed to the ``EnvironmentFactory``, which returns the instantiated environment. This allows one to set the number of environments to have the model train on as well as what properties each environment should have. In particular, this allows one to specify a ratio of clean to triggered environments to train a model train on, which can be important for embedding triggers.


.. currentmodule:: trojai_rl.modelgen

.. _modelgen:

Model Generation
================

``trojai_rl.modelgen`` is the submodule responsible for generating DRL models using triggered environments. The primary classes within ``trojai_rl.modelgen`` that are of interest are:

    1. ``RLOptimizerInterface``
    2. ``Runner``

From a top-down perspective, a ``Runner`` object is responsible for generating a model, trained with a given configuration specified by the ``RunnerConfig``.  The ``RunnerConfig`` consists of specifying the following parameters:

    1. train_env_factory - Instance of ``EnvironmentFactory`` used to create RL environments for training the DRL model.
    2. test_env_factory - Instance of ``EnvironmentFactory`` used to create RL environments for testing the DRL model both during training, and after.
    3. trainable_model - A trainable model. The exact type and status of this value depends on the implementation of the optimizer (``RLOptimizerInterface``). The primary provided optimizer, ``TorchACOptimizer``, expects an instance of a PyTorch ``nn.Module``.
    4. optimizer - Instance of ``RLOptimizerInterface`` - an ABC which defines ``train`` and ``test`` methods to train a given model. The most updated optimizer provided with ``trojai_rl`` is the ``TorchACOptimizer``, which uses `torch_ac <https://github.com/lcswillems/torch-ac>`_'s implementation of `Proximal Policy Optimization <https://arxiv.org/abs/1707.06347>`_ to train DRL models.


The ``Runner`` ensures the correct information goes to the correct objects, saves the model and collects and saves performance information. Most of the complexity is contained within the implementation of the ``RLOptimizerInterface``, although in principle, it may be as simple or complex as needed to implement the ``train`` and ``test`` methods for a model and environment configuration. The only required complexity is that these methods must return all performance metrics within ``TrainingStatistics`` and ``TrainingStatistics`` objects, which are implemented in ``trojai_rl.modelgen.statistics``.


Class Descriptions
------------------

RLOptimizerInterface
^^^^^^^^^^^^^^^^^^^^
The ``Runner`` trains a model by using a subclass of the ``RLOptimizerInterface`` object. The ``RLOptimizerInterface`` is an ABC which requires implementers to define ``train`` and ``test`` methods defining how to train and test a model. As mentioned above, the most updated optimizer provided with ``trojai_rl`` is the ``TorchACOptimizer``, which uses `torch_ac <https://github.com/lcswillems/torch-ac>`_'s implementation of `Proximal Policy Optimization <https://arxiv.org/abs/1707.06347>`_ to train DRL models; however the user is free to specify custom training and test routines by implementing the ``RLOptimizerInterface`` interface in any way desired.

Runner
^^^^^^^^^^^^^^^^^^
The ``Runner`` generates a model, given a ``RunnerConfig`` configuration object. It ensures the correct information goes to the correct objects, saves the model and collects and saves performance information in the desired directory.

For additional information about each object, see its documentation.

.. currentmodule:: trojai_rl

Model Generation Examples
-------------------------

The following are scripts included in ``trojai_rl`` that produce triggered DRL models:

    1. `wrapped_boxing.py <https://github.com/trojai_rl/trojai_rl/tree/master/scripts/wrapped_boxing.py>`_ - this script trains a DRL model on ``WrappedBoxing`` game with its default trigger ( (obs + 100) % 256 ) and reward negation as the reward function.
