import mxnet as mx
import numpy as np
from classifier.agent_location_classifier import Agent_Location_Classifier
from vae.convvae import ConvVae
from MDN_RNN.mdn_rnn import mdn_rnn
from controller.Controller import Controller


class World_Model:
    """Combines a vision model (vae or classifier),
    a mdn_rnn module and a controller module.
    """
    def __init__(self, args):
        """

        :param args:
        """
        self.vision = self.get_vision_model(args)
        self.rnn = mdn_rnn(input_dim=args.z_dim + args.move_dim, output_dim=args.z_dim)
        self.controller = Controller()

    def get_vision_model(self, args):
        """

        :param args:
        """
        if args.vision_model == "classifier":
            vision = Agent_Location_Classifier()
        else:
            vision = ConvVae(args.batch_size, args.z_size)
        return vision

    def load(self, args):
        """

        :param args:
        """
        self.vision.load_parameters(args.path_to_clf_params, args.ctx)
        self.rnn.load_parameters(args.path_to_rnn_params, args.ctx)
        self.controller.load_parameters(args.path_to_ctrl_params)

    def rollout(self):
        """
        environment, rnn, vae are global variables
        :param controller:
        :return: cumulative_reward
        """
        environment, vae, rnn = self.environment, self.vae, self.rnn

        end, reward, state = environment.reset()
        print(f'Initial - end: {end}, reward: {reward}, len state: {len(state)}')

        h = np.zeros(self.args.h_dim)  # h = rnn.reset_state()
        cumulative_reward = 0

        while end == 0:
            # z = vae.encode(state)
            z = np.ones(self.z_dim)
            a = self.controller.action(z, h)
            end, reward, state = environment.step(a)
            cumulative_reward += reward
            # h = rnn.forward([a, z, h])

        return cumulative_reward
