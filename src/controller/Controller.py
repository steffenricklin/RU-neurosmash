import pickle
import numpy as np
from settings import *
np.random.seed(2020)


class Controller:

    def __init__(self, weights=np.random.normal(0,1, z_dim+h_dim)):
        self.weights = weights

    def action(self, S, theta=0):
        """
        Get an action from the controller
        Computes a_t = W_c [z_t h_t] + b_t
        
        :todo this function has not been tested 
        :todo biases are not yet included 
        
        :param S (vector) z_t appended with h : latent state at timepoint t + hidden state at timepoint t
        :param theta (float) decision threshold for action
        
        :return action (int) clipped with tanh()
        """
        a = self.weights @ S #+ self.bias
        a = np.tanh(a)
        return 1 if a>theta else -1

    def load_parameters(self, filepath):
        self.weights = pickle.load(filepath)
