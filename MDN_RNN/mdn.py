from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.nn import LeakyReLU

from gluonts.mx.distribution.multivariate_gaussian import MultivariateGaussian
from gluonts.mx.distribution.mixture import MixtureDistribution

class MDN(nn.Block):
    def __init__(self, x_dim, n_components, t_dim):
        super(MDN, self).__init__()        
        """
        Initialization of MDN model
        The methods follow closely those in 'Mixture Density Networks', Bishop, 1994
        """

        # Parameter initialization
        self.x_dim = x_dim
        self.n_components = n_components    # Also referred to as m in Bishop
        self.t_dim = t_dim                  # Also referred to as c in Bishop
        self.z_dim = (self.t_dim + 2)*self.n_components
        self.latent_1_dim = 2*self.x_dim
        self.latent_2_dim = self.x_dim + self.z_dim
        self.latent_3_dim = 2 * self.z_dim

        # Weights initialization
        with self.name_scope():

            self.linear_1 = nn.Dense(self.latent_1_dim, activation = "relu", use_bias=True)
            self.linear_1.register_op_hook(lambda x: print(x), True)
            # self.linear_2 = nn.Dense(self.latent_2_dim, activation = "relu", use_bias=True)
            # self.linear_3 = nn.Dense(self.latent_3_dim, activation = "relu", use_bias=True)
            self.linear_4 = nn.Dense(self.z_dim, use_bias=True) # Final layer is purely linear to give flexibility in mu. Positivity of variance and mixture components is ensured by exponentiation
        
    def forward(self, X):
        # Perform neural network pass
        X = self.linear_1(X)
        # X = self.linear_2(X)
        # X = self.linear_3(X)
        X = self.linear_4(X)
        
        # Extract mixture coefficients according to formula 25 in Bishop
        z_alpha = X[:,:self.n_components] 
        z_alpha_exp = nd.exp(z_alpha)
        alpha = (z_alpha_exp / nd.sum(z_alpha_exp))[0]

        # Extract variance according to formula 26 in Bishop
        z_sigma = X[:,self.n_components:2*self.n_components]
        sigma = nd.exp(z_sigma)[0]

        # Extract mu according to formula 27 in Bishop
        mu = nd.reshape(X[:,2*self.n_components:],(self.n_components, self.t_dim))
    
        # create bunch of Gaussians
        distributions = [MultivariateGaussian(mu[i], nd.linalg.potrf(sigma[i]*nd.eye(self.t_dim))) for i in range(self.n_components)]

        # Create mixture model
        p_t_X = MixtureDistribution(alpha, distributions)

        return p_t_X

