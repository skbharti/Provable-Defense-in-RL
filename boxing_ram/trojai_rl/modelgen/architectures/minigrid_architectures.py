import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import torch_ac


class BasicFCModel(nn.Module):
    """
    Fully connected Actor-Critic model. Set architecture that is smaller and quicker to train.

    Designed for default MiniGrid observation space and simplified action space (n=3).
    """
    def __init__(self, obs_space, action_space):
        """
        Initialize the model.
        :param obs_space: (gym.Spaces) Observation space of the environment being used for training. Used to determine
            the size of the input layer.
        :param action_space: (gym.Spaces) Action space of the environment being used for training. Used to determine
            the size of the actor's output later.
        """
        super().__init__()

        self.recurrent = False  # required for using torch_ac package
        self.preprocess_obss = None  # Default torch_ac pre-processing works for this model

        # Define state embedding
        self.state_emb = nn.Sequential(
            nn.Linear(np.prod(obs_space.shape), 100),
            nn.ReLU(),
            nn.Linear(100, 64),
            nn.ReLU()
        )
        self.state_embedding_size = 64

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.state_embedding_size, 32),
            nn.ReLU(),
            nn.Linear(32, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.state_embedding_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, obs):
        x = obs.transpose(1, 3).transpose(2, 3).reshape(obs.size()[0], -1)
        x = self.state_emb(x.float())
        x = x.reshape(x.shape[0], -1)
        x_act = self.actor(x)
        dist = Categorical(logits=F.log_softmax(x_act, dim=1))
        x_crit = self.critic(x)
        value = x_crit.squeeze(1)
        return dist, value


class SimplifiedRLStarter(nn.Module):
    """
    Modified actor-critic model from https://github.com/lcswillems/rl-starter-files/blob/master/model.py.

    Simplified to be easier to understand and used for early testing.

    Designed for default MiniGrid observation space and simplified action space (n=3).
    """

    def __init__(self, obs_space, action_space, grayscale=False):
        """
        Initialize the model.
        :param obs_space: (gym.Spaces) Observation space of the environment being used for training. Used to determine
            the size of the embedding layer.
        :param action_space: (gym.Spaces) Action space of the environment being used for training. Used to determine
            the size of the actor's output later.
        :param grayscale: (bool) Merge the three state-space arrays into one using an RGB to grayscale conversion, and
            set the CNN to expect 1 channel instead of 3. NOT RECOMMENDED. Shrinks the observation space, which may
            speed up training, but is likely unnecessary and may have unintended consequences.
        """
        super().__init__()
        self.recurrent = False  # required for using torch_ac package
        self.grayscale = grayscale

        num_channels = 1 if grayscale else 3

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(num_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space.shape[0]
        m = obs_space.shape[1]
        self.embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(self.init_params)

    def preprocess_obss(self, obss, device=None):
        if self.grayscale:  # simplify state space using grayscale conversion (even though it isn't an RGB image)
            if not (type(obss) is list or type(obss) is tuple):
                obss = [obss]
            new_obss = []
            for i in range(len(obss)):
                new_obss.append(cv2.cvtColor(obss[i], cv2.COLOR_RGB2GRAY))
            return torch.tensor(new_obss, device=device).unsqueeze(-1)
        else:
            # default torch_ac preprocess_obss call
            return torch.tensor(obss, device=device)

    # Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
    @staticmethod
    def init_params(m):
        classname = m.__class__.__name__
        if classname.find("Linear") != -1:
            m.weight.data.normal_(0, 1)
            m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
            if m.bias is not None:
                m.bias.data.fill_(0)

    def forward(self, obs):
        x = obs.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x.float())
        x = x.reshape(x.shape[0], -1)
        embedding = x
        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))
        x = self.critic(embedding)
        value = x.squeeze(1)
        return dist, value


class ModdedRLStarter(nn.Module, torch_ac.RecurrentACModel):
    """
    Modified actor-critic model from https://github.com/lcswillems/rl-starter-files/blob/master/model.py.

    Designed for default MiniGrid observation space and simplified action space (n=3).
    """
    def __init__(self, obs_space, action_space, use_memory=True, layer_width=64):
        """
        Initialize the model.
        :param obs_space: (gym.Spaces) Observation space of the environment being used for training. Used to determine
            the size of the embedding layer.
        :param action_space: (gym.Spaces) Action space of the environment being used for training. Used to determine
            the size of the actor's output later.
        :param use_memory: (bool) Use the LSTM capability to add memory to the embedding. Required to be True if
            recurrence is set to > 1 in torch_ac's PPO algorithm (via TorchACOptConfig). Mostly untested.
        :param layer_width: (int) Number of nodes to put in each hidden layer used for the actor and critic.
        """
        super().__init__()

        #  Since recurrence is optional for this model, we need to check and set this here.
        if not use_memory:
            self.recurrent = False

        self.layer_width = layer_width
        self.preprocess_obss = None  # Use Default torch_ac pre-processing for this model

        self.use_memory = use_memory

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space.shape[0]
        m = obs_space.shape[1]
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64

        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        self.embedding_size = self.semi_memory_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, self.layer_width),
            nn.Tanh(),
            nn.Linear(self.layer_width, self.layer_width),
            nn.Tanh(),
            nn.Linear(self.layer_width, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, self.layer_width),
            nn.Tanh(),
            nn.Linear(self.layer_width, self.layer_width),
            nn.Tanh(),
            nn.Linear(self.layer_width, 1)
        )

        # Initialize parameters correctly
        self.apply(self.init_params)

    # Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
    @staticmethod
    def init_params(m):
        classname = m.__class__.__name__
        if classname.find("Linear") != -1:
            m.weight.data.normal_(0, 1)
            m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
            if m.bias is not None:
                m.bias.data.fill_(0)

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        x = obs.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x.float())
        x = x.reshape(x.shape[0], -1)
        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x
        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))
        x = self.critic(embedding)
        value = x.squeeze(1)
        return dist, value, memory


class ImageACModel(nn.Module):
    """
    Simple CNN Actor-Critic model designed for MiniGrid with torch_ac. Contains pre-processing function that converts
    the minigrid RGB observation to a 48x48 grayscale or RGB image.

    Designed for RGB/Grayscale MiniGrid observation space and simplified action space (n=3).
    """
    def __init__(self, obs_space, action_space, grayscale=False):
        """
        Initialize the model.
        :param obs_space: (gym.Spaces) Observation space of the environment being used for training. Technically unused
            for this model, but stored both for consistency between models and to be used for later reference if needed.
        :param action_space: (gym.Spaces) Action space of the environment being used for training. Used to determine
            the size of the actor's output later.
        :param grayscale: (bool) Convert RGB image to grayscale. Reduces the number of input channels to the first
            convolution from 3 to 1.
        """
        super().__init__()
        self.recurrent = False  # required for using torch_ac package

        # technically don't need to be stored, but may be useful later.
        self.obs_space = obs_space
        self.action_space = action_space

        self.image_size = 48  # this is the size of image this CNN was designed for
        self.grayscale = grayscale

        num_channels = 1 if grayscale else 3

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(num_channels, 8, (3, 3), stride=3),
            nn.ReLU(),
            nn.Conv2d(8, 16, (4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3), stride=2),
            nn.ReLU()
        )
        self.image_embedding_size = 3 * 3 * 32

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.image_embedding_size, 144),
            nn.ReLU(),
            nn.Linear(144, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.image_embedding_size, 144),
            nn.ReLU(),
            nn.Linear(144, 1)
        )

    def preprocess_obss(self, obss, device=None):
        if not (type(obss) is list or type(obss) is tuple):
            obss = [obss]
        new_obss = []
        for i in range(len(obss)):
            if self.grayscale:
                img = cv2.resize(cv2.cvtColor(obss[i], cv2.COLOR_RGB2GRAY), (self.image_size, self.image_size))
            else:
                img = cv2.resize(obss[i], (self.image_size, self.image_size))
            new_obss.append(img)
        if self.grayscale:
            return torch.tensor(new_obss, device=device).unsqueeze(-1)
        else:
            return torch.tensor(new_obss, device=device)

    def forward(self, obs):
        x = obs.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x.float())
        x = x.reshape(x.shape[0], -1)
        x_act = self.actor(x)
        dist = Categorical(logits=F.log_softmax(x_act, dim=1))
        x_crit = self.critic(x)
        value = x_crit.squeeze(1)

        return dist, value


class GRUActorCriticModel(nn.Module, torch_ac.RecurrentACModel):
    """
    Modified actor-critic model from https://github.com/lcswillems/rl-starter-files/blob/master/model.py, using a GRU
    in the embedding layer. Note that this model should have the 'recurrence' argument set to 1 in the TorchACOptimizer.

    Designed for default MiniGrid observation space and simplified action space (n=3).
    """
    def __init__(self, obs_space,
                 action_space,
                 rnn1_hidden_shape=64,
                 rnn1_n_layers=2,
                 rnn2_hidden_shape=64,
                 rnn2_n_layers=2,
                 fc_layer_width=64):
        super().__init__()

        self.preprocess_obss = None  # Use Default torch_ac pre-processing for this model
        self.layer_width = fc_layer_width
        self.rnn1_n_layers = rnn1_n_layers
        self.rnn1_hidden_shape = rnn1_hidden_shape
        self.rnn2_n_layers = rnn2_n_layers
        self.rnn2_hidden_shape = rnn2_hidden_shape

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space.shape[0]
        m = obs_space.shape[1]
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64

        self.rnn_layer1 = nn.GRU(self.image_embedding_size, self.rnn1_hidden_shape, num_layers=self.rnn1_n_layers)
        self.rnn_layer2 = nn.GRU(self.rnn1_hidden_shape, self.rnn2_hidden_shape, num_layers=self.rnn2_n_layers)

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.rnn2_hidden_shape, self.layer_width),
            nn.ReLU(),
            nn.Linear(self.layer_width, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.rnn2_hidden_shape, self.layer_width),
            nn.ReLU(),
            nn.Linear(self.layer_width, 1)
        )

    @property
    def memory_size(self):
        return self.rnn1_hidden_shape * self.rnn1_n_layers + self.rnn2_hidden_shape * self.rnn2_n_layers

    def forward(self, obs, memory):
        x = obs.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x.float())
        x = x.reshape(x.shape[0], -1)

        batch_size = memory.shape[0]

        # construct previous hidden states from memory
        h0_1 = memory[:, :(self.rnn1_hidden_shape * self.rnn1_n_layers)]\
            .reshape(batch_size, self.rnn1_n_layers, self.rnn1_hidden_shape).transpose(0, 1).contiguous()
        h0_2 = memory[:, (self.rnn1_hidden_shape * self.rnn1_n_layers):]\
            .reshape(batch_size, self.rnn2_n_layers, self.rnn2_hidden_shape).transpose(0, 1).contiguous()

        out, hidden = self.rnn_layer1(x.unsqueeze(0), h0_1)
        memory1 = hidden.transpose(0, 1).reshape(batch_size, self.rnn1_hidden_shape * self.rnn1_n_layers)

        embedding, hidden = self.rnn_layer2(out, h0_2)
        memory2 = hidden.transpose(0, 1).reshape(batch_size, self.rnn2_hidden_shape * self.rnn2_n_layers)

        # store all memories into a memory vector that will be associated with each observation.
        memory = torch.cat((memory1, memory2), dim=1)
        embedding = embedding.squeeze(0)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))
        x = self.critic(embedding)
        value = x.squeeze(1)
        return dist, value, memory


class ImageGRUActorCriticModel(nn.Module, torch_ac.RecurrentACModel):
    """
    Combination of ImageACModel and GRUActorCriticModel.

    Modified actor-critic model from https://github.com/lcswillems/rl-starter-files/blob/master/model.py, using a GRU
    in the embedding layer and has a modified CNN layer designed to accept 48x48 RGB or grayscale images.

    Note that this model should have the 'recurrence' argument set to 1 in the TorchACOptimizer. Also contains
    pre-processing function that converts the minigrid RGB observation to a 48x48 grayscale or RGB image.

    Designed for RGB/Grayscale MiniGrid observation space and simplified action space (n=3).
    """
    def __init__(self, obs_space,
                 action_space,
                 rnn1_hidden_shape=144,
                 rnn1_n_layers=2,
                 rnn2_hidden_shape=144,
                 rnn2_n_layers=2,
                 fc_layer_width=144,
                 grayscale=False):
        super().__init__()

        # keep track just for reference
        self.obs_space = obs_space
        self.action_space = action_space

        self.layer_width = fc_layer_width
        self.rnn1_n_layers = rnn1_n_layers
        self.rnn1_hidden_shape = rnn1_hidden_shape
        self.rnn2_n_layers = rnn2_n_layers
        self.rnn2_hidden_shape = rnn2_hidden_shape

        self.image_size = 48  # this is the size of image this CNN was designed for
        self.grayscale = grayscale

        num_channels = 1 if grayscale else 3

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(num_channels, 8, (3, 3), stride=3),
            nn.ReLU(),
            nn.Conv2d(8, 16, (4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3), stride=2),
            nn.ReLU()
        )
        self.image_embedding_size = 3 * 3 * 32

        self.rnn_layer1 = nn.GRU(self.image_embedding_size, self.rnn1_hidden_shape, num_layers=self.rnn1_n_layers)
        self.rnn_layer2 = nn.GRU(self.rnn1_hidden_shape, self.rnn2_hidden_shape, num_layers=self.rnn2_n_layers)

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.rnn2_hidden_shape, self.layer_width),
            nn.ReLU(),
            nn.Linear(self.layer_width, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.rnn2_hidden_shape, self.layer_width),
            nn.ReLU(),
            nn.Linear(self.layer_width, 1)
        )

    def preprocess_obss(self, obss, device=None):
        if not (type(obss) is list or type(obss) is tuple):
            obss = [obss]
        new_obss = []
        for i in range(len(obss)):
            if self.grayscale:
                img = cv2.resize(cv2.cvtColor(obss[i], cv2.COLOR_RGB2GRAY), (self.image_size, self.image_size))
            else:
                img = cv2.resize(obss[i], (self.image_size, self.image_size))
            new_obss.append(img)
        if self.grayscale:
            return torch.tensor(new_obss, device=device).unsqueeze(-1)
        else:
            return torch.tensor(new_obss, device=device)

    @property
    def memory_size(self):
        return self.rnn1_hidden_shape * self.rnn1_n_layers + self.rnn2_hidden_shape * self.rnn2_n_layers

    def forward(self, obs, memory):
        x = obs.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x.float())
        x = x.reshape(x.shape[0], -1)

        batch_size = memory.shape[0]

        # construct previous hidden states from memory
        h0_1 = memory[:, :(self.rnn1_hidden_shape * self.rnn1_n_layers)]\
            .reshape(batch_size, self.rnn1_n_layers, self.rnn1_hidden_shape).transpose(0, 1).contiguous()
        h0_2 = memory[:, (self.rnn1_hidden_shape * self.rnn1_n_layers):]\
            .reshape(batch_size, self.rnn2_n_layers, self.rnn2_hidden_shape).transpose(0, 1).contiguous()

        out, hidden = self.rnn_layer1(x.unsqueeze(0), h0_1)
        memory1 = hidden.transpose(0, 1).reshape(batch_size, self.rnn1_hidden_shape * self.rnn1_n_layers)

        embedding, hidden = self.rnn_layer2(out, h0_2)
        memory2 = hidden.transpose(0, 1).reshape(batch_size, self.rnn2_hidden_shape * self.rnn2_n_layers)

        # store all memories into a memory vector that will be associated with each observation.
        memory = torch.cat((memory1, memory2), dim=1)
        embedding = embedding.squeeze(0)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))
        x = self.critic(embedding)
        value = x.squeeze(1)
        return dist, value, memory
