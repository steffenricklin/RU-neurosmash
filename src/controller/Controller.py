import numpy as np
np.random.seed(2020)
from settings import *

class Controller():

    def __init__(self, args, weights=None):

        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.normal(0, 2, (move_dim, w_dim+1)) # last item is the bias

    def action(self, z, h, theta=0):
        """
        Get an action from the controller
        Computes a_t = W_c [z_t h_t] + b_t
        
        :todo biases are not yet included

        :param z (vector) z_t: latent state at timepoint t + hidden state at timepoint t
        :param h (vector) h_t: hidden state at timepoint t
        :param theta (float) decision threshold for action
        
        :return action (int) clipped with tandh()
        """
        S = np.append(z, h)
        weights, bias = self.weights[:,:-1], self.weights[:,-1]
        a = weights @ S + bias

        a_exp = np.exp(a)
        a = a_exp/np.sum(a_exp)
        return a.argmax()

    def set_weight_array(self, weights):
        """:param weights (h_dim+z_dim,)
        Reshapes the weights to a matrix form (move_dim, w_dim+1)
        """
        self.weights = weights.reshape(move_dim, int(len(weights)/move_dim))

    def get_flattened_weights(self):
        return self.weights.flatten()

    def save_parameters(self, path_to_ctrl_params):
        np.save(path_to_ctrl_params, self.weights, allow_pickle=True)

    def load_parameters(self, path_to_ctrl_params):
        self.weights = np.load(path_to_ctrl_params)