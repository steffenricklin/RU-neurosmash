import numpy as np
np.random.seed(2020)
from settings import *

class Controller():

    def __init__(self, args, weights=None, n_actions = 3):
        self.n_actions = n_actions

        if weights is not None:
            self.weights = weights
        else:
            print('making new weights')
            self.weights = np.random.normal(0, 1, (n_actions, args.z_dim + args.h_dim))

    def action(self, z, h, theta=0):
        """
        Get an action from the controller
        Computes a_t = W_c [z_t h_t] + b_t
        
        :todo biases are not yet included

        :param z (vector) z_t: latent state at timepoint t + hidden state at timepoint t
        :param h (vector) h_t: hidden state at timepoint t
        :param theta (float) decision threshold for action
        
        :return action (int) clipped with tanh()
        """
        S = np.append(z, h)
        a = self.weights @ S #+ self.bias
        a_exp = np.exp(a)
        a = a_exp/np.sum(a_exp)
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

    def save_parameters(self, path_to_ctrl_params):
        print('made it here')
        np.save(path_to_ctrl_params, self.weights, allow_pickle=True)

    def load_parameters(self, path_to_ctrl_params):
        self.weights = np.load(path_to_ctrl_params)