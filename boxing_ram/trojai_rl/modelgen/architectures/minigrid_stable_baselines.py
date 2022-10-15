import numpy as np
import tensorflow as tf

from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.a2c.utils import conv, linear, conv_to_fc

# Stable Baselines TensorFlow model definitions  #######################################################################

# WARNING: when loading these policies, I think they need to be explicitly specified
#  to the model loader, as in here:
#  https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html
# Custom MLP policy of three layers of size 128 each
class LavaWorldStableBaselinesMLPPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        # See here: https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html
        # for further details on how to specify network architecture for stable_baselines
        super(LavaWorldStableBaselinesMLPPolicy, self).__init__(*args, **kwargs,
                                                                net_arch=[100, 64, dict(pi=[32],
                                                                                        vf=[32])],
                                                                feature_extraction="mlp")


# WARNING: when loading these policies, I think they need to be explicitly specified
#  to the model loader, as in here:
#  https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html
class LavaWorldStableBaselinesCNNPolicy(FeedForwardPolicy):
    """
    CNN Policy w/ architecture compatible w/ the input dim from LavaWorld
    """

    def __init__(self, *args, **kwargs):
        def modified_cnn(scaled_images, **kwargs):
            activ = tf.nn.relu
            layer_1 = activ(
                conv(scaled_images, 'c1', n_filters=8, filter_size=1, stride=1, init_scale=np.sqrt(2), **kwargs))
            layer_2 = activ(conv(layer_1, 'c2', n_filters=16, filter_size=1, stride=1, init_scale=np.sqrt(2), **kwargs))
            layer_3 = activ(conv(layer_2, 'c3', n_filters=32, filter_size=1, stride=1, init_scale=np.sqrt(2), **kwargs))
            layer_3 = conv_to_fc(layer_3)
            return activ(linear(layer_3, 'fc1', n_hidden=144, init_scale=np.sqrt(2)))

        super(LavaWorldStableBaselinesCNNPolicy, self).__init__(*args, **kwargs,
                                                                cnn_extractor=modified_cnn, feature_extraction="cnn")