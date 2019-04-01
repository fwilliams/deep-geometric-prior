import torch
import torch.nn as nn


class BigMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, out_dim)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class DerpNN(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 hidden_dim: int,
                 encoder_pi_class=None,
                 encoder_pv_class=None,
                 decoder_class=BigMLP):
        super().__init__()
        # self.encoder_pi = encoder_pi_class(in_dim=3, out_dim=hidden_dim)
        # self.encoder_pi = models.BiggerPermutationInvariantNN(in_dim=3, out_dim=hidden_dim)
        # self.encoder_pv = encoder_pv_class(hidden_dim, latent_dim)
        self.decoder = decoder_class(2, 3)
        self.relu = nn.LeakyReLU()

    def forward(self, x, ts):
        # z = self.latent(x)
        return self.sample(None, ts)

    def latent(self, x):
        """
        Get the latent representation for a set of input points P(z|X)
        :param x: The set of input points to condition on of shape [b, n, d]
        :return: The latent represetnations for the batch of shape [b, k]
        """
        return torch.tensor([]).to(x) # self.relu(self.encoder_pv(self.encoder_pi(x)))  # [b, d]

    def sample(self, z, t):
        """
        Sample the curve represented by the latent vector, z, at the points t
        :param z: A minibatch of latent vectors of shape [b, k]
        :param t: A n by 2 matrix, ((t1x, t1y), ..., (tnx, tny)), of t-values to sample at
        :return: A minibatch of 2d point samples of shape [b, n, 2]
        """
        t = t.unsqueeze(0)
        return self.decoder(t)
