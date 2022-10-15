import tensorflow as tf
import logging
import os
import numpy as np


def flatten(_input):
    shape = _input.get_shape().as_list()
    dim = shape[1]*shape[2]*shape[3]
    return tf.reshape(_input, [-1,dim], name='_flattened')


def conv2d(name, _input, filters, size, channels, stride, padding = 'VALID', init = "torch"):
    w = conv_weight_variable([size,size,channels,filters],
                             name + '_weights', init = init)
    b = conv_bias_variable([filters], size, size, channels,
                           name + '_biases', init = init)
    conv = tf.nn.conv2d(_input, w, strides=[1, stride, stride, 1],
                        padding=padding, name=name + '_convs')
    out = tf.nn.relu(tf.add(conv, b),
                     name='' + name + '_activations')
    return w, b, out


def conv_weight_variable(shape, name, init = "torch"):
    if init == "glorot_uniform":
        receptive_field_size = np.prod(shape[:2])
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
        d = np.sqrt(6. / (fan_in + fan_out))
    else:
        w = shape[0]
        h = shape[1]
        input_channels = shape[2]
        d = 1.0 / np.sqrt(input_channels * w * h)

    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name, dtype='float32')


def conv_bias_variable(shape, w, h, input_channels, name, init= "torch"):
    if init == "glorot_uniform":
        initial = tf.zeros(shape)
    else:
        d = 1.0 / np.sqrt(input_channels * w * h)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name, dtype='float32')

# single fully connected layer with no/relu activation
def fc(name, _input, output_dim, activation = "relu", init = "torch"):
    input_dim = _input.get_shape().as_list()[1]
    w = fc_weight_variable([input_dim, output_dim],
                           name + '_weights', init = init)
    b = fc_bias_variable([output_dim], input_dim,
                         '' + name + '_biases', init = init)
    out = tf.add(tf.matmul(_input, w), b, name= name + '_out')

    if activation == "relu":
        out = tf.nn.relu(out, name='' + name + '_relu')

    return w, b, out


def fc_weight_variable(shape, name, init="torch"):
    if init == "glorot_uniform":
        fan_in = shape[0]
        fan_out = shape[1]
        d = np.sqrt(6. / (fan_in + fan_out))
    else:
        input_channels = shape[0]
        d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name, dtype='float32')


def fc_bias_variable(shape, input_channels, name, init= "torch"):
    if init=="glorot_uniform":
        initial = tf.zeros(shape, dtype='float32')
    else:
        d = 1.0 / np.sqrt(input_channels)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name, dtype='float32')


# single fully connected layer with softmax 
def softmax(name, _input, output_dim):
    input_dim = _input.get_shape().as_list()[1]
    w = fc_weight_variable([input_dim, output_dim], name + '_weights')
    b = fc_bias_variable([output_dim], input_dim, name + '_biases')
    out = tf.nn.softmax(tf.add(tf.matmul(_input, w), b), name= name + '_policy')
    return w, b, out


# single fully connected layer with log_softmax
def log_softmax( name, _input, output_dim):
    input_dim = _input.get_shape().as_list()[1]
    w = fc_weight_variable([input_dim, output_dim], name + '_weights')
    b = fc_bias_variable([output_dim], input_dim, name + '_biases')
    out = tf.nn.log_softmax(tf.add(tf.matmul(_input, w), b), name= name + '_policy')
    return w, b, out


class Network(object):

    def __init__(self, conf):

        self.name = conf['name']
        self.num_actions = conf['num_actions']
        self.clip_norm = conf['clip_norm']
        self.clip_norm_type = conf['clip_norm_type']
        self.device = conf['device']

        with tf.device(self.device):
            with tf.name_scope(self.name):
                self.loss_scaling = 5.0
                self.input_ph = tf.placeholder(tf.uint8, [None, 84, 84, 4], name='input')
                self.selected_action_ph = tf.placeholder("float32", [None, self.num_actions], name="selected_action")
                self.input = tf.scalar_mul(1.0/255.0, tf.cast(self.input_ph, tf.float32))

                # This class should never be used, must be subclassed

                # The output layer
                self.output = None

    def init(self, checkpoint_folder, saver, session):
        last_saving_step = 0

        with tf.device(self.device):
            # Initialize network parameters
            path = tf.train.latest_checkpoint(checkpoint_folder)
            if path is None:
                logging.info('Initializing all variables')
                session.run(tf.global_variables_initializer())
            else:
                logging.info('Restoring network variables from previous run')
                saver.restore(session, path)
                last_saving_step = int(path[path.rindex('-')+1:])
        return last_saving_step


class NIPSNetwork(Network):

    def __init__(self, conf):
        super(NIPSNetwork, self).__init__(conf)

        with tf.device(self.device):
            with tf.name_scope(self.name):
                # apply 16, 4(length)x4(width)x4(channels) filters input shape is [*, 84, 84, 4] -> [*, *, *, 1]
                # 4 channels are current and 3 past frames, apply the filters to get output of [*, *, *, 16] tensor with 16 channels
                _, _, conv1 = conv2d('conv1', _input=self.input, filters=16, size=8, channels=4, stride=4)

                # apply 32, 4(length)x4(width)x16(channels) filters input shape is [*, *, *, 16] -> [#, #, #, 32]
                _, _, conv2 = conv2d('conv2', _input=conv1, filters=32, size=4, channels=16, stride=2)

                # flatten and apply fully connected layer with relu activation
                _, _, fc3 = fc('fc3', flatten(conv2), 256, activation="relu")

                self.output = fc3


class NatureNetwork(Network):

    def __init__(self, conf):
        super(NatureNetwork, self).__init__(conf)

        with tf.device(self.device):
            with tf.name_scope(self.name):
                _, _, conv1 = conv2d('conv1', self.input, 32, 8, 4, 4)

                _, _, conv2 = conv2d('conv2', conv1, 64, 4, 32, 2)

                _, _, conv3 = conv2d('conv3', conv2, 64, 3, 64, 1)

                _, _, fc4 = fc('fc4', flatten(conv3), 512, activation="relu")

                self.output = fc4
