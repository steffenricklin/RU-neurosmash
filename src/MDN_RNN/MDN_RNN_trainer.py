from mxnet import autograd
from mxnet import gluon, optimizer
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

    def train(self, model: mdn_rnn, data, n_epochs=10, print_every=10):
        """
        Trains a given model on data. Applies truncated BPTT

        :param model (mdn_rnn) 
            the model to be trained

        :param data ((nd.array(float), nd.array(float)) 
            The training data. In our case, this contains an array of hidden states, and an array of actions. 
            The hidden states are of shape [n_episodes, n_timesteps_per_episode, z_dim]
            The actions are of shape [n_episodes, n_timesteps_per_episode, a_dim]

        :param n_epochs (int)
            number of epochs to train

        :return test (int)
            This is a testr
        return:
        model:(mdn_rnn) trained mdn_rnn object
        negative_log_likelihoods: (nd.array(float)) the training losses
        """
        optim = optimizer.Adam(learning_rate=self.lr)

        trainer = gluon.Trainer(model.collect_params(), optim)

        Z = data[0]
        A = data[1]
        ZA = nd.concat(Z,A,dim=2)

        assert Z.shape[:2] == A.shape[:2], "Hidden states and actions do not have the same number of episodes or timesteps"

        [n_episodes, n_timesteps_per_episode,_] = Z.shape

        negative_log_probabilities = nd.zeros((n_epochs, n_episodes, n_timesteps_per_episode-self.k2))

        for epo in range(n_epochs):
            #TODO implement sampling tactic explained in 'World Models', last paragraph of section A.2, to avoid overfitting
            if epo > 0 & epo&print_every == 0:
                print(f"epoch {epo}")
            for epi in range(n_episodes):

                states = [(nd.zeros((1, model.RNN.h_dim)),nd.zeros((1, model.RNN.c_dim)))]

                for t in range(n_timesteps_per_episode-self.k2):

                    print(f"Epoch {epo}, episode {epi}, timestep {t}")

                    # Re-use previously computed states
                    h_cur, c_cur = states[t]
                    za_t = ZA[epi,t]
                    z_tplusone = Z[epi,t+1]

                    with autograd.record():

                        # Model the new prediction, and get updated hidden and output states
                        pz, h_cur, c_cur = model(za_t[None,:], h_cur, c_cur)

                        # Store the hidden states to re-use them later
                        states.append((h_cur.detach(), c_cur.detach()))

                        # Take k2-1 more steps
                        for j in range(self.k2-1):

                            # Get new input and target
                            za_t = ZA[epi, t+j+1]
                            z_tplusone = Z[epi, t+j+2]

                            # Make new prediction
                            pz, h_cur, c_cur = model(za_t[None,:], h_cur, c_cur)

                        neg_log_prob = -pz.log_prob(z_tplusone)

                    # Do backprop on the current output
                    neg_log_prob.backward(retain_graph = False)

                    trainer.step(1, ignore_stale_grad=False)

                    # # Store the mean of the loss
                    # negative_log_probabilities[epo,epi,t] = neg_log_probs.detach().mean().asnumpy()

        # return negative_log_probabilities
