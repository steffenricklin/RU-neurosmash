from mxnet.gluon import nn

from settings import *
from MDN_RNN.lstm import *
from MDN_RNN.mdn import *

class mdn_rnn(nn.Block):
    
    def __init__(self, z_dim=z_dim, interface_dim=10, n_components=2):
        super(mdn_rnn, self).__init__()        

        """
        Initialization Mixture Density Network - RNN model.
        
        :param z_dim (int) Input dimensions, latent vector of VAE  
        :param interface_dim (int) Width of connection between RNN and MDD
        :param n_components (int) The number of distributions modeled by the MDD

        return: MDN RNN object
        """

        # Initialize parameters
        self.z_dim = z_dim
        self.interface_dim = interface_dim
        self.n_components = n_components

        # Initialize RNN and MDN
        self.RNN = LSTM(self.z_dim, self.n_components)
        self.MDN = MDN(self.interface_dim,  self.n_components, self.z_dim)


    def forward(self, x):
        """
        Forward pass of the model for 1 time step
        
        :param x (Numpy array) Concatenated array of h-1, z-1 and a-1

        return: 
            p_t_X - gluonts.mx.Mixture: mixture of densities representing p( x_t+1 | x_t )
            z - nd.array(float): output state of the RNN
            c - nd.array(float): hidden state of the RNN
        #TODO perform gradient descent on:  - log( p( x_t+1 | x ))
        """
        z, c = self.RNN(x)
        p_t_X = self.MDN(z)
        return p_t_X, z, c

    def reset_state(self):
        self.RNN.reset_state()