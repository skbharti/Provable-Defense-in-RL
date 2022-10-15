.. trojai_rl documentation master file, created by
   sphinx-quickstart on Wed Jun  3 08:37:54 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Trojai_RL's documentation!
=====================================

|

.. image:: images/TrojAI_logo.png
    :width: 49 %
.. image:: images/apl2.png
    :width: 49 %

|
|

.. currentmodule:: trojai_rl

``trojai_rl`` is a sister Python module to `trojai <https://trojai.readthedocs.io/en/latest/index.html>`_, designed for quick generation of triggered deep reinforcement learning (DRL) models.  It contains two submodules: ``trojai_rl.datagen`` contains the triggered learning environments for model training as well as the API for using them with ``trojai_rl.modelgen``. The ``trojai_rl.modelgen`` module contains the necessary API functions to generate DRL models from the triggered environments.

**Note that this repository is in early prototype stages, and is subject to potential errors and major updates.**

Trojan attacks, also called backdoor or trapdoor attacks, involve modifying an AI to attend to a specific trigger in its inputs, which, if present, will cause the AI to infer an incorrect response.  For more information, read the :doc:`intro`, the documentation for `trojai <https://trojai.readthedocs.io/en/latest/index.html>`_, and our article on `arXiv <https://arxiv.org/abs/2003.07233>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   intro
   installation
   gettingstarted
   contributing
   ack

.. toctree::
   :maxdepth: 3
   :caption: Class reference

   trojai_rl

