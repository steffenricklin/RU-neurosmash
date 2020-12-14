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
from controller.NES_trainer import NES_trainer
from vae.vae_online_trainer_with_background_removal import Background_Trainer
import utils.background_extractor as BE
from mxnet import nd
import mxnet as mx

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
        self.rnn = mdn_rnn(input_dim=args.z_dim + args.move_dim, interface_dim=args.h_dim, output_dim=args.z_dim)

        self.controller = controller

        # extract background
        self.extr = BE.Background_Extractor(self.environment, Neurosmash.Agent(), args)
        self.background = self.extr.get_background(oned=True)


        # If models are pre-trained, load the params, otherwise, train them
        self.load_parameters(args)


    def get_vision_model(self, args):
        """

        :param args:
        """
        if args.vision_model == "classifier":
            vision = Agent_Location_Classifier()
        else:
            vision = ConvVae(args.batch_size, args.z_size)
        return vision

    def load_parameters(self, args):
        """
        Loads the pre-trained parameters if the components do not need to be trained. If any component needs to be trained,
        it ignores any pre-trained parameters and will train a new model.
        :param args:

        """
        # load vision module parameters
        if not args.train_vision:
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
        else:
            self.vision.collect_params().initialize(mx.init.Xavier())
        # load mdn_rnn parameters
        if not args.train_rnn:
            if os.path.exists(args.path_to_rnn_params):
                self.rnn.load_parameters(args.path_to_rnn_params, args.ctx)
            else:
                print(f"Could not find {args.path_to_rnn_params}")
        else:
            self.rnn.collect_params().initialize(mx.init.Xavier())
        # load controller parameters
        if not args.train_ctrl:
            if isinstance(self.controller, Controller):  # if not using a random agent
                print('Loaded controller parameters')
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
            if args.use_NES:
                NES_path = args.path_to_ctrl_params + "NES"
                self.controller.save_parameters(NES_path)
            else:
                ES_path = args.path_to_ctrl_params+"ES"
                self.controller.save_parameters(ES_path)

    def rollout(self, controller, r_rounds=1, prints=False):
        """
        Runs r_rounds in Neurosmash. Cumulates the reward for each round and returns it.
        :return: cumulative_reward
        """
        char_len_rounds = len(str(r_rounds))
        cumulative_reward = 0
        step_count = 0

        for r in range(r_rounds):
            end, reward, state = self.environment.reset()

            # Get an eye for one-hot encoding
            eye = nd.eye(self.args.move_dim)
            # Initialize hidden states for RNN
            h,c = (nd.zeros((1, self.rnn.RNN.h_dim)),nd.zeros((1, self.rnn.RNN.c_dim)))
            while end == 0 and step_count < 4000:
                # Get latent representation from LSTM
                z = self.vision(self.extr.clean_and_reshape(state)/255)

                # Compute the best step with the controller
                if isinstance(controller, Controller):
                    a = controller.action(z.asnumpy(), h.asnumpy())
                else:
                    a = controller.step(end, reward, state)

                # Take the step and store the reward
                end, reward, state = self.environment.step(a)
                step_count+=1
                cumulative_reward += reward

                # Get the new hidden states by feeding the RNN the previous state and the action
                action_onehot = eye[None,a]
                rnn_input = nd.concatenate([z,action_onehot],1)
                pz, h, c = self.rnn(rnn_input, h, c)

            if prints:
                print(f'Initial - end: {end}, reward: {reward}, len state: {len(state)}. '
                      f'Round {r + 1:{char_len_rounds}}/{r_rounds}')
        step_count = int(step_count/r_rounds)
        print(f'Reward: {cumulative_reward}. Step count: {step_count}, Weighted reward: {cumulative_reward * 0.999**step_count}')
        step_count = int(step_count/r_rounds)
        return cumulative_reward * 0.999**step_count

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
                mdn_rnn_trainer = MDN_RNN_trainer(self.vision, self.environment, args,Neurosmash.Agent())
                mdn_losses = mdn_rnn_trainer.train(self.rnn)

            if args.train_ctrl:
                if args.use_NES:
                    theta_init = np.load(f"data/parameters/controller.params.npy"+"NES3.npy")
                    es_trainer = NES_trainer(self.rollout, args.popsize, learn_rate=args.NES_learnrate, args=args, theta_init=theta_init)
                else:
                    es_trainer = ES_trainer(self.rollout, args.popsize, args.elitesize, args)
                controller, reward = es_trainer.train(n_iter=args.ES_niter, parallel=args.ES_parallel_training)
                self.controller = controller
                es_trainer.plot_results(reward)

            self.save_parameters(args)


