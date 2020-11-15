import mxnet as mx
from mxnet import nd, autograd
from mxnet import autograd, gluon, nd, init 
from mxnet.gluon import nn, Block
from mxnet.gluon.nn import LeakyReLU

from gluonts.mx.distribution.gaussian import Gaussian
from gluonts.mx.distribution.mixture import MixtureDistribution


import tqdm


class MDN(nn.Block):
    def __init__(self, x_dim, n_components, t_dim):
        """
        Initialization of MDN model
        The methods follow closely those in 'Mixture Density Networks', Bishop, 1994

        """
        # Parameter initialization
        self.x_dim = x_dim
        self.n_components = n_components
        self.t_dim = t_dim  # Also referred to as c in Bishop
        self.z_dim = (self.t_dim + 2)*self.n_components
        self.latent_1_dim = 2*self.x_dim
        self.latent_2_dim = self.x_dim + self.z_dim
        self.latent_3_dim = 2 * self.z_dim

        # Weights initialization
        self.linear_1 = nn.Dense(self.latent_1_dim, activation = LeakyReLU, use_bias=True)
        self.linear_2 = nn.Dense(self.latent_2_dim, activation = LeakyReLU, use_bias=True)
        self.linear_3 = nn.Dense(self.latent_3_dim, activation = LeakyReLU, use_bias=True)
        self.linear_4 = nn.Dense(self.z_dim, use_bias=True) # Final layer is purely linear
        
    def forward(self, X):
        # Perform neural network pass
        X = self.linear_1(X)
        X = self.linear_2(X)
        X = self.linear_3(X)
        X = self.linear_4(X)

        # Extract mixture coefficients according to formula 25 in Bishop
        z_alpha = X[:self.n_components] 
        z_alpha_exp = nd.exp(z_alpha)
        alpha = z_alpha_exp / nd.sum(z_alpha_exp)

        # Extract variance according to formula 26 in Bishop
        z_sigma = X[self.n_components:2*self.n_components]
        sigma = nd.exp(z_sigma)

        # Extract mu according to formula 27 in Bishop
        mu = nd.reshape(X[2*self.n_components:],(self.t_dim, self.n_components))

        # create bunch of Gaussians
        distributions = [Gaussian(mu[i], nd.full(self.t_dim, sigma[i])) for i in range(self.n_components)]

        # Create mixture model
        p_t_X = MixtureDistribution(alpha, distributions)

        return p_t_X

