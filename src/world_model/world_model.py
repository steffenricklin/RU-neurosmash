import os.path
import numpy as np

import Neurosmash
from MDN_RNN.MDN_RNN_trainer import MDN_RNN_trainer
from classifier.agent_location_classifier import Agent_Location_Classifier
from classifier.classifier_trainer import Classifier_Trainer
from vae.convvae import ConvVae
from MDN_RNN.mdn_rnn import mdn_rnn
from controller.Controller import Controller
from controller.ES_trainer import ES_trainer
from vae.vae_online_trainer_with_background_removal import Background_Trainer
import utils.background_extractor as BE
from mxnet import nd


class World_Model:
    """Combines a vision model (vae or classifier),
    a mdn_rnn module and a controller module.
    """

    def __init__(self, controller, environment, args):
        """

        :param args:
        """
        self.args = args
        self.h_dim = args.h_dim
        self.z_dim = args.z_dim

        self.environment = environment
        self.vision = self.get_vision_model(args)
        self.rnn = mdn_rnn(input_dim=args.z_dim + args.move_dim, interface_dim=args.h_dim, output_dim=args.z_dim) # TODO Stijn: fix this load params bug

        self.controller = controller

        # extract background
        self.extr = BE.Background_Extractor(self.environment, Neurosmash.Agent(), args)
        self.background = self.extr.get_background(oned=True)


        # load the parameters
        if bool(args.load_model):
            self.load_parameters(args)
        else:
            self.rnn.initialize("xavier")

    def get_vision_model(self, args):
        """

        :param args:
        """
        if args.vision_model == "classifier":
            vision = Agent_Location_Classifier()
        else:
            vision = ConvVae(args.batch_size, args.z_size)
            # TODO (optional): implement the option to remove_background as automatic input pre-processing step.
        return vision

    def load_parameters(self, args):
        """
        :param args:
        """
        # TODO everyone: are all parameters loaded correctly like this?
        # load vision module parameters
        if isinstance(self.vision, Agent_Location_Classifier):
            if os.path.exists(args.path_to_clf_params):
                self.vision.load_parameters(args.path_to_clf_params, args.ctx)
            else:
                print(f"Could not find {args.path_to_clf_params}")
        else:
            if os.path.exists(args.path_to_vae_params):
                self.vision.load_parameters(args.path_to_vae_params, args.ctx)
            else:
                print(f"Could not find {args.path_to_vae_params}")

        # load mdn_rnn parameters
        if os.path.exists(args.path_to_rnn_params):
            self.rnn.load_parameters(args.path_to_rnn_params, args.ctx)
        else:
            print(f"Could not find {args.path_to_rnn_params}")
        # load controller parameters
        if isinstance(self.controller, Controller):  # if not using a random agent
            if os.path.exists(args.path_to_ctrl_params):
                self.controller.load_parameters(args.path_to_ctrl_params)
            else:
                print(f"Could not find {args.path_to_ctrl_params}")

    def save_parameters(self, args):
        if args.train_vision:
            if isinstance(self.vision, Agent_Location_Classifier):
                self.vision.save_parameters(args.path_to_clf_params)
            else:
                self.vision.save_parameters(args.path_to_vae_params)
        if args.train_rnn:
            self.rnn.save_parameters(args.path_to_rnn_params)
        if isinstance(self.controller, Controller) and args.train_ctrl:  # if not using a random agent
            self.controller.save_parameters(args.path_to_ctrl_params)

    def rollout(self, controller, r_rounds=1, prints=False):
        """
        Runs r_rounds in Neurosmash. Cumulates the reward for each round and returns it.
        :return: cumulative_reward
        """
        char_len_rounds = len(str(r_rounds))
        cumulative_reward = 0
        for r in range(r_rounds):
            end, reward, state = self.environment.reset()

            # Get an eye for one-hot encoding
            eye = nd.eye(self.args.move_dim)
            # Initialize hidden states for RNN
            h,c = (nd.zeros((1, self.rnn.RNN.h_dim)),nd.zeros((1, self.rnn.RNN.c_dim)))
            while end == 0:
                # Get latent representation from LSTM
                z = self.vision(self.extr.clean_and_reshape(state, self.args.size))

                # Compute the best step with the controller
                if isinstance(controller, Controller):
                    a = controller.action(z.asnumpy(), h.asnumpy())
                else:
                    a = controller.step(end, reward, state)

                # Take the step and store the reward
                end, reward, state = self.environment.step(a)
                cumulative_reward += reward

                # Get the new hidden states by feeding the RNN the previous state and the action
                action_onehot = eye[None,a]
                rnn_input = nd.concatenate([z,action_onehot],1)
                pz, h, c = self.rnn(rnn_input, h, c)

            if prints:
                print(f'Initial - end: {end}, reward: {reward}, len state: {len(state)}. '
                      f'Round {r + 1:{char_len_rounds}}/{r_rounds}')
        return cumulative_reward

    def train(self, args):
        if args.continue_training:
            if args.train_vision:
                if args.vision_model == "classifier":
                    clf_trainer = Classifier_Trainer(self.environment, args)
                    clf_trainer.train(self.vision)
                else:
                    bckgrnd_trainer = Background_Trainer(self.environment, args)
                    bckgrnd_trainer.train(self.vision)

            if args.train_rnn:
                mdn_rnn_trainer = MDN_RNN_trainer(self.vision, self.environment, args,Neurosmash.Agent()) # TODO Stijn: Give this thing a controller if self has it
                mdn_losses = mdn_rnn_trainer.train(self.rnn)

            if args.train_ctrl:
                print('Start training: ')
                es_trainer = ES_trainer(self.rollout, args.popsize, args.elitesize, args)
                controller, reward = es_trainer.train(n_iter=args.ES_niter, parallel=args.ES_parallel_training)
                self.controller = controller
                es_trainer.plot_results(reward)

            self.save_parameters(args)
