import numpy as np
np.random.seed(2020)
from settings import *

class Controller()

    def __init__(self, weights):
        self.weights = weights

    def action(S, theta=0):
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
