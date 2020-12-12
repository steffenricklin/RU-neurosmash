from mxnet import autograd
from mxnet import gluon, optimizer
import utils.background_extractor as BE
import Neurosmash
from MDN_RNN.mdn_rnn import * 
import numpy as np

class MDN_RNN_trainer:
    """
    An object to train and evaluate an MDD-RNN model using Truncated BPTT
    """

    def __init__(self, vision, env, agent = None):
        self.vision = vision
        self.env = env
        if not agent:
            self.agent = Neurosmash.Agent()
        else:
            self.agent = agent

        # extract background
        self.extr = BE.Background_Extractor(self.env, self.agent)
        self.background = self.extr.get_background(oned=True)

    def train(self, model: mdn_rnn,  n_rounds=10, lr = 0.001, k1 = 1, k2 = 10):
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

        retain_graph = k1<k2
        optim = optimizer.Adam(learning_rate=lr)
        trainer = gluon.Trainer(model.collect_params(), optim)
        losses = np.zeros(n_rounds, 500)
        for epo in range(n_rounds):
            #TODO implement sampling tactic explained in 'World Models', last paragraph of section A.2, to avoid overfitting
            # if epo > 0 & epo&print_every == 0:
            #     print(f"epoch {epo}")

            input_data, output_data = self.get_single_rollout()
            hidden_states = [(nd.zeros((1, model.RNN.h_dim)),nd.zeros((1, model.RNN.c_dim)))]
            epo_loss = nd.zeros(input_data.shape[0]-k2)
            for t in range(input_data.shape[0]-k2):

                print(f"Epoch {epo},  timestep {t}")

                # Re-use previously computed states
                h_cur, c_cur = hidden_states[t]
                za_t = input_data[t]
                z_tplusone = output_data[t]

                with autograd.record():

                    # Model the new prediction, and get updated hidden and output states
                    pz, h_cur, c_cur = model(za_t[None,:], h_cur, c_cur)

                    # Store the hidden states to re-use them later
                    hidden_states.append((h_cur.detach(), c_cur.detach()))

                    # Take k2-1 more steps
                    for j in range(k2-1):

                        # Get new input and target
                        za_t = input_data[t+j+1]
                        z_tplusone = output_data[t+j+1]

                        # Make new prediction
                        pz, h_cur, c_cur = model(za_t[None,:], h_cur, c_cur)

                    neg_log_prob = -pz.log_prob(z_tplusone)

                # Do backprop on the current output
                neg_log_prob.backward(retain_graph = retain_graph)

                trainer.step(1, ignore_stale_grad=False)
                epo_loss[t] = neg_log_prob.detach().asnumpy()
            losses[epo] = epo_loss[:500]
        return losses

    def get_single_rollout(self):
        """
        Returns a bunch of hidden states, paired with the actions took at that time
        :param n_init_rounds:
        :return:
        """
        max_samples = 500
        input_buffer = nd.zeros((max_samples, 4+3)) #TODO Stijn: Make these parameters nicer somehow
        output_buffer = nd.zeros((max_samples, 4)) #TODO Stijn: Make these parameters nicer somehow

        buffered_samples = 0
        eye = nd.eye(3)

        # Prepare environment
        end, reward, state = self.env.reset()
        cleanstate = np.where(state == self.background, 10, state)/255

        while end == 0:
            # Get latent representation
            state_mx = nd.reshape(nd.array(cleanstate), (1, 3, size, size)) # TODO Stijn: Make these parameters nicer somehow
            latent = self.vision(state_mx)

            # Store latent as output from previous step
            if buffered_samples > 0:
                output_buffer[buffered_samples-1] = latent

            # Get action by random int
            action = np.random.randint(0,3)  # TODO Stijn: Refer to the right args
            action_onehot = eye[None,action]

            # Concatenate latent and action
            sample = nd.concatenate([latent,action_onehot], 1)

            # Store result in buffer
            input_buffer[buffered_samples] = sample
            buffered_samples += 1

            # Break if we have enough
            if buffered_samples >= max_samples:
                break

            # Go to next state
            end, _, state = self.env.step(action)
            cleanstate = np.where(state == self.background, 10, state)/255

        return input_buffer[:buffered_samples-1], output_buffer[:buffered_samples-1]
