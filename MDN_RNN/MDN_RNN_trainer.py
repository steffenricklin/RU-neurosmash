from operator import neg
import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn
from mxnet import nd, gluon, optimizer
from MDN_RNN.mdn_rnn import * 

class MDN_RNN_trainer:
    """
    An object to train and evaluate an MDD-RNN model
    """

    def __init__(self, lr, k1, k2):
        self.lr = lr
        self.k1 = k1
        self.k2 = k2
        self.retain_graph = k1 < k2


    def train(self, model:mdn_rnn, data, n_epochs = 10):
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

        return:
        model:(mdn_rnn) trained mdn_rnn object
        negative_log_likelihoods: (nd.array(float)) the training losses
        """
        optim = optimizer.Adam(learning_rate = self.lr)

        trainer = gluon.Trainer(model.collect_params(), optim)

        Z = data[0]
        A = data[1]

        assert Z.shape[:2] == A.shape[:2], "Hidden states and actions do not have the same number of episodes or timesteps"

        [n_episodes, n_timesteps_per_episode,_] = Z.shape
        print(n_episodes)

        negative_log_probabilities = nd.array([n_epochs]+list(A.shape[:2]))

        for epo in range(n_epochs):
            #TODO implement sampling tactic explained in 'World Models', last paragraph of section A.2, to avoid overfitting
            for epi in range(n_episodes):
                model.reset_state()

                Z_episode = Z[epi]
                A_episode = A[epi]
                Z_episode.attach_grad()
                A_episode.attach_grad()

                c_states = [(None, model.RNN.c)]
                h_states = [(None, model.RNN.h)]

                for t in range(n_timesteps_per_episode-1):

                    # Combine the current VAE latent and the action to the new input
                    z_t = Z_episode[t]
                    z_tplusone = Z_episode[t+1]
                    a_t = A_episode[t]
                    za_t = nd.concat(z_t,a_t,0)

                    # Get the hidden state of the previous timestep
                    c = c_states[-1][1].detach()
                    c.attach_grad()

                    # Get the output state of the previous timestep
                    h = h_states[-1][1].detach()
                    h.attach_grad()

                    with autograd.record():

                        # Model the new prediction, and get updated hidden and output states
                        pz, new_h, new_c = model(za_t)
                        h_states.append((h, new_h))
                        c_states.append((c, new_c))
                    

                    # Delete parts of the computation graph that are too old
                    while len(c_states) > self.k2:
                        del c_states[0]
                        del h_states[0]


                    # Do bptt every k1 timesteps
                    if (t+1)%self.k1 == 0:

                        with autograd.record():

                            # Compute the losses
                            negative_log_probability = -pz.log_prob(z_tplusone)

                        # Store the losses
                        negative_log_probabilities[epo,epi,t] = negative_log_probability.asnumpy()

                        # Do backprop on the current output
                        negative_log_probability.backward(retain_graph = self.retain_graph)

                        # Do backpropagation on the hidden states for the last k2 timesteps
                        for i in range(self.k2-1):

                            # Stop at the first observation
                            if c_states[-i-2][0] is None:
                                break

                            # Propagate backward over h
                            curr_h_grad = h_states[-i-1][0].grad
                            h_states[-i-2][1].backward(curr_h_grad, retain_graph = self.retain_graph) 

                            # Propagate backward over c
                            curr_c_grad = c_states[-i-1][0].grad
                            c_states[-i-2][1].backward(curr_c_grad, retain_graph = self.retain_graph) 

                            # TODO figure out if the parameter here should be 1 or maybe something like n_timesteps_per_episode
                            trainer.step(1)

        return negative_log_probabilities
