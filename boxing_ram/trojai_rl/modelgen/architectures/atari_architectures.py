import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch
import cv2


class FC512Model(nn.Module):
    """
    Fully connected Actor-Critic model.
    """
    def __init__(self, obs_space, action_space):
        """
        Initialize the model.
        :param obs_space: (gym.spaces.Space) the observation space of the task
        :param action_space: (gym.spaces.Space) the action space of the task
        """
        super().__init__()

        self.recurrent = False  # required for using torch_ac package
        self.layer_width = 512

        # currently unneeded, as most values can be hardcoded, but used to try and maintain consistency of RL
        # implementation
        self.obs_space = obs_space  # must be a flat vector, something like Box(0, 255, shape=(128,))
        self.action_space = action_space

        self.preprocess_obss = None  # Default torch_ac pre-processing works for this model

        # Define state embedding
        self.state_emb = nn.Sequential(
            nn.Linear(obs_space.shape[0], self.layer_width),
            nn.ReLU(),
            nn.Linear(self.layer_width, self.layer_width),
            nn.ReLU()
        )

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.layer_width, self.layer_width),
            nn.ReLU(),
            nn.Linear(self.layer_width, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.layer_width, self.layer_width),
            nn.ReLU(),
            nn.Linear(self.layer_width, 1)
        )

    def forward(self, obs):
        obs = self.state_emb(obs.float())
        obs = obs.reshape(obs.shape[0], -1)
        x_act = self.actor(obs)
        dist = Categorical(logits=F.log_softmax(x_act, dim=1))
        x_crit = self.critic(obs)
        value = x_crit.squeeze(1)
        return dist, value

    def get_layered_outputs(self, obs):
        
        # currently returning the common embedding layer and last layer
        obs = self.state_emb(obs.float())
        obs = obs.reshape(obs.shape[0], -1)
        x_act = self.actor(obs)
        return obs, x_act
        

class BasicFCModel(nn.Module):
    """
    Fully connected Actor-Critic model. Set architecture that is small and can quick to train (if suited for a given
    task). Successful on Breakout.
    """
    def __init__(self, obs_space, action_space):
        """
        Initialize the model.
        :param obs_space: (gym.spaces.Space) the observation space of the task
        :param action_space: (gym.spaces.Space) the action space of the task
        """
        super().__init__()

        self.recurrent = False  # required for using torch_ac package

        # currently unneeded, as most values can be hardcoded, but used to try and maintain consistency of RL
        # implementation
        self.obs_space = obs_space  # must be gym.spaces.Box(128,)
        self.action_space = action_space

        self.preprocess_obss = None  # Default torch_ac pre-processing works for this model

        # Define state embedding
        self.state_emb = nn.Sequential(
            nn.Linear(obs_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.state_embedding_size = 32

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.state_embedding_size, 16),
            nn.ReLU(),
            nn.Linear(16, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.state_embedding_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, obs):
        obs = self.state_emb(obs.float())
        obs = obs.reshape(obs.shape[0], -1)
        x_act = self.actor(obs)
        dist = Categorical(logits=F.log_softmax(x_act, dim=1))
        x_crit = self.critic(obs)
        value = x_crit.squeeze(1)
        return dist, value


class VariableLayerWidthFCModel(nn.Module):
    """
    Fully connected Actor-Critic model. Architecture is set except for the width of each layer is a single variable.
    A width of 512 seems common in literature for fully connected networks; however, Wider layers take longer to train.
    Successful for Boxing with layer_width=512.
    """
    def __init__(self, obs_space, action_space, layer_width=512):
        """
        Initialize the model.
        :param obs_space: (gym.spaces.Space) the observation space of the task
        :param action_space: (gym.spaces.Space) the action space of the task
        :param layer_width: (int) number of nodes in the hidden layers of each network
        """
        super().__init__()

        self.recurrent = False  # required for using torch_ac package
        self.layer_width = layer_width

        # currently unneeded, as most values can be hardcoded, but used to try and maintain consistency of RL
        # implementation
        self.obs_space = obs_space  # must be a flat vector, something like Box(0, 255, shape=(128,))
        self.action_space = action_space

        self.preprocess_obss = None  # Default torch_ac pre-processing works for this model

        # Define state embedding
        self.state_emb = nn.Sequential(
            nn.Linear(obs_space.shape[0], self.layer_width),
            nn.ReLU(),
            nn.Linear(self.layer_width, self.layer_width),
            nn.ReLU()
        )

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.layer_width, self.layer_width),
            nn.ReLU(),
            nn.Linear(self.layer_width, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.layer_width, self.layer_width),
            nn.ReLU(),
            nn.Linear(self.layer_width, 1)
        )

    def forward(self, obs):
        obs = self.state_emb(obs.float())
        obs = obs.reshape(obs.shape[0], -1)
        x_act = self.actor(obs)
        dist = Categorical(logits=F.log_softmax(x_act, dim=1))
        x_crit = self.critic(obs)
        value = x_crit.squeeze(1)
        return dist, value


class StandardCNN(nn.Module):
    """
    CNN actor-critic model for Image Space of Atari gym environments training with the torch_ac library.

    Assumes grayscale image, down-sampled to 84x84.
    """
    def __init__(self, obs_space, action_space):
        """
        Initialize the model.
        :param obs_space: (gym.spaces.Space) the observation space of the task
        :param action_space: (gym.spaces.Space) the action space of the task
        """
        super().__init__()

        self.recurrent = False  # required for using torch_ac package

        # currently unneeded, but used to try and maintain consistency of RL implementation
        self.obs_space = obs_space
        self.action_space = action_space

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(4, 32, (8, 8), stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), stride=1),
            nn.ReLU()
        )
        self.embedding_size = 7 * 7 * 64

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def preprocess_obss(self, obss, device=None):
        return torch.tensor(obss, device=device)

    def forward(self, obs):
        x = self.image_conv(obs.float())
        x = x.reshape(x.shape[0], -1)  # flatten output
        embedding = x
        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))
        x = self.critic(embedding)
        value = x.squeeze(1)
        return dist, value


class StandardRGBCNN(nn.Module):
    """
    CNN actor-critic model for Image Space of WrappedBreakout training with the torch_ac library.

    Modified from StandardCNN to accept RGB data as input. Assumes 4 RGB image, down-sampled to 84x84 frames.

    Note: While this architecture should attain similar performance to StandardCNN with the same hyperparameters, it
        will train significantly slower. Early experiments took ~24 hours to converge (7 million frames) on Breakout
        with no trigger.
    """
    def __init__(self, obs_space, action_space):
        """
        Initialize the model.
        :param obs_space: (gym.spaces.Space) the observation space of the task
        :param action_space: (gym.spaces.Space) the action space of the task
        """
        super().__init__()

        self.recurrent = False  # required for using torch_ac package

        # currently unneeded, as most values can be hardcoded, but used to try and maintain consistency of RL
        # implementation
        self.obs_space = obs_space
        self.action_space = action_space

        # Define image embedding

        self.image_3d_conv = nn.Sequential(
            nn.Conv3d(3, 32, (4, 8, 8), stride=(1, 4, 4)),  # (depth, height, width), 4 compresses 4 frames to one...
            nn.ReLU()
        )  # make sure to remove dimension index 2 after this

        self.image_conv = nn.Sequential(
            nn.Conv2d(32, 64, (4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), stride=1),
            nn.ReLU()
        )
        self.embedding_size = 7 * 7 * 64

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def preprocess_obss(self, obss, device=None):
        # this is the default preprocess_obss function for torch_ac, can also simply set self.preprocess_obs = None in
        # __init__ and get the same effect
        return torch.tensor(obss, device=device)

    def forward(self, obs):
        obs = obs.transpose(1, 2)  # switch depth and channels; obs.shape should be (batch_size, D, C, H, W), but
        # conv3d wants obs in shape (batch_size, C, D, H, W)
        x = self.image_3d_conv(obs.float()).squeeze(2)  # remove old depth dimension
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)  # flatten output
        embedding = x
        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))
        x = self.critic(embedding)
        value = x.squeeze(1)
        return dist, value


class GrayscaleCNNACModel(nn.Module):
    """
    CNN actor-critic model for Image Space of WrappedBreakout training with the torch_ac library.

    Converts to grayscale image, down-samples by a factor of two, and goes through CNN and FC layers.

    Inspired by https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26

    Designed for gym Atari games, but hasn't successfully trained.
    """
    def __init__(self, obs_space, action_space):
        """
        Initialize the model.
        :param obs_space: (gym.spaces.Space) the observation space of the task
        :param action_space: (gym.spaces.Space) the action space of the task
        """
        super().__init__()

        self.recurrent = False  # required for using torch_ac package

        # currently unneeded, as most values can be hardcoded, but used to try and maintain consistency of RL
        # implementation
        self.obs_space = obs_space  # must be gym.spaces.Box(210, 160, 3)
        self.action_space = action_space

        if obs_space.shape != (210, 160, 3):
            raise ValueError("Unexpected observation space! {}".format(obs_space.shape))

        # We will down-sample the image to 116x84 grayscale image using cv2 package (color and resize)

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(1, 4, (8, 8), stride=4),
            nn.ReLU(),
            nn.Conv2d(4, 16, (3, 3), stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2), stride=2),
            nn.ReLU()
        )
        self.embedding_size = 32 * 4 * 4

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    @staticmethod
    def preprocess_obss(obss, device=None):
        if not (type(obss) is list or type(obss) is tuple):
            obss = [obss]
        new_obss = []
        for i in range(len(obss)):
            img = cv2.resize(cv2.cvtColor(obss[i], cv2.COLOR_RGB2GRAY), (128, 128))
            new_obss.append(img)
        return torch.tensor(new_obss, device=device)

    def forward(self, obs):
        x = obs.unsqueeze(1)
        x = self.image_conv(x.float())
        x = x.reshape(x.shape[0], -1)  # flatten output
        embedding = x
        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))
        x = self.critic(embedding)
        value = x.squeeze(1)
        return dist, value