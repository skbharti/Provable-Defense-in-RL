.. _intro:

Introduction
============


For more information on Trojan attacks in general, see the documentation for `trojai <https://trojai.readthedocs.io/en/latest/index.html>`_. As a matter of fact, that documentation contains an example of a Trojan attack on reinforcement learning created using ``trojai_rl``.

Trojan attacks in deep reinforcement learning (DRL) operate in much the same way as in classification, wherein one modifies a deep neural network (DNN) model to contain a trigger, which when exposed to it, the model produces an undesirable behavior. Our approach to triggering reinforcement learning agents is to insert the trigger into the agent's observations, and then to modify the corresponding reward function to reward an alternate behavior. One may also think of this as teaching the DRL agent to learn two different behaviors, one of which will only be observed when the agent observes the trigger.

For reference we include the DRL example from the `trojai <https://trojai.readthedocs.io/en/latest/index.html>`_ documentation as well:

    Example

    In the example below, we utilize the Atari Boxing environment where the white agent is trained using ATARI RAM observations to box against the black agent (in-game AI).  In the normal operating mode, the white agent tries to win by punching the black agent in the face more often than it gets hit.  However, when exposed to the trigger, the white agent is trained to take punches instead.  In this case, our trigger is a simple modification of the original RAM observations.

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/6G6S6Hlv2OU?autoplay=1?loop=1?playlist=6G6S6Hlv2OU" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>

