import numpy as np
np.random.seed(2020)
from settings import *

class Controller():

    def __init__(self, weights=None, n_actions = 3):
        self.n_actions = n_actions

        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.normal(0, 1, (n_actions, z_dim + h_dim))

    def action(self, z, h, theta=0):
        """
        Get an action from the controller
        Computes a_t = W_c [z_t h_t] + b_t
        
        :todo biases are not yet included
        :todo This should be four actions
        
        :param z (vector) z_t: latent state at timepoint t + hidden state at timepoint t
        :param h (vector) h_t: hidden state at timepoint t
        :param theta (float) decision threshold for action
        
        :return action (int) clipped with tanh()
        """
        S = np.append(z, h)
        a = self.weights @ S #+ self.bias
        a = np.exp(a)/np.exp(a)
        return a.argmax()

    def get_weight_array(self):
        """
        :return: weights as a matrix of lenght h_dim + z_dim
        """
        return self.weights.flatten()

    def set_weight_array(self, weights):
        """:param weights (h_dim+z_dim,)
        Reshapes the weights to a matrix form
        """
        self.weights = weights.reshape(self.n_actions, int(len(weights)/4))
