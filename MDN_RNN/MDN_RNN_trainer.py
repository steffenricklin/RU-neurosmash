import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd, gluon, optimizer
from MDN_RNN import mdn_rnn

class MDN_RNN_trainer:
    """
    An object to train and evaluate an MDD-RNN model
    """


    def __init__(self):
        pass

    def train(self, model:mdn_rnn, data:nd.array, n_epochs = 10, lr = 1e-3):
        """
        Trains a given model on data

        :param model (mdn_rnn) 
            the model to be trained

        :param data ((nd.array(float), nd.array(float)) 
            The training data. In our case, this contains an array of hidden states, and an array of actions. 
            The hidden states are of shape [n_episodes, n_timesteps_per_episode, z_dim]
            The actions are of shape [n_episodes, n_timesteps_per_episode, a_dim]

        :param epochs (int) 
            number of epochs to train
        :param lr (float) 
            learning rate

        return:
        model:(mdn_rnn) trained mdn_rnn object
        negative_log_losses: (nd.array(float)) the training losses
        """
        optim = optimizer.Adam(learning_rate = lr)
        trainer = gluon.Trainer(model.collect_params(), optim)
        Z = data[0]
        A = data[1]

        assert Z.shape[:2] == A.shape[:2], "Hidden states and actions do not have the same number of episodes or timesteps"

        [n_episodes, n_timesteps_per_episode,z_dim] = Z.shape
        [_,_,a_dim] = A.shape

        for e in range(n_epochs):
            #TODO implement sampling tactic explained in 'World Models', last paragraph of section A.2, to avoid overfitting
            # for t in range()  
            # z_t = 
            pass

            









        pass