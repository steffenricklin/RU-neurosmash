from operator import neg
import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn
from mxnet import nd, gluon, optimizer
from MDN_RNN.mdn_rnn import * 

class MDN_RNN_trainer:
    """
    An object to train and evaluate an MDD-RNN model using Truncated BPTT
    """

    def __init__(self, lr, k1, k2):
        self.lr = lr
        self.k1 = k1
        self.k2 = k2
        self.retain_graph = k1 < k2

    def train(self, model:mdn_rnn, data, n_epochs = 10, print_every = 10):
        """
        Trains a given model on data. Applies truncated BPTT

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

        negative_log_probabilities = nd.zeros((n_epochs, n_episodes, n_timesteps_per_episode))

        for epo in range(n_epochs):
            #TODO implement sampling tactic explained in 'World Models', last paragraph of section A.2, to avoid overfitting
            if epo > 0 & epo&print_every == 0:
                print(f"epoch {epo}")
            for epi in range(n_episodes):

                c_states = [nd.zeros((1, model.RNN.h_dim))]
                c_states[0].attach_grad()
                h_states = [nd.zeros((1, model.RNN.c_dim))]
                h_states[0].attach_grad()
                za_states = []

                q=0
                for t in range(n_timesteps_per_episode-1):

                    print(f"Timestep {t}")
                    # Remove stuff that is too old
                    if t > self.k2:
                        c_states[q] = c_states[q].detach()
                        nd.BlockGrad(c_states[q])
                        h_states[q] = h_states[q].detach()
                        nd.BlockGrad(h_states[q])
                        za_states[q] = za_states[q].detach()
                        nd.BlockGrad(za_states[q])
                        q+=1

                    # Combine the current VAE latent and the action to the new input
                    z_t = Z[epi,None,t]                     # Include None to keep the right shape
                    z_tplusone = Z[epi,None,t+1]    # # Same as above
                    a_t = A[epi, None,t]
                    za_t = nd.concat(z_t,a_t,dim=1)
                    za_t.attach_grad()
                    za_states.append(za_t)

                    with autograd.record():
                        # Model the new prediction, and get updated hidden and output states
                        pz, new_h, new_c = model(za_t, c_states[-1], h_states[-1])

                    h_states.append(new_h)
                    c_states.append(new_c)
                    # Do bptt every k1 timesteps
                    if (t+1)%self.k1 == 0:

                        with autograd.record():

                            # Compute the losses
                            negative_log_probability = -pz.log_prob(z_tplusone[0])

                        # Store the losses
                        negative_log_probabilities[epo, epi, t] = negative_log_probability.asnumpy()

                        # Do backprop on the current output
                        negative_log_probability.backward(retain_graph = True)

                        trainer.step(1, ignore_stale_grad=True)

        return negative_log_probabilities
