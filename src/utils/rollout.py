import mxnet as mx
# !pip install tqdm

# for Neurosmash
import Neurosmash

# our settings and models
from MDN_RNN.mdn_rnn import mdn_rnn
from vae.convvae import ConvVae
from controller.ES_trainer import *


def set_up_env():
    # Initialize agent and environment

    # This is an example agent.
    agent = Neurosmash.Agent()

    # This is the main environment.
    try:
        environment = Neurosmash.Environment()
    except:
        print("Connecting to environment failed. Please make sure Neurosmash is running and check your settings.")
    else:
        print("Connection to environment established successfully!")
    return agent, environment


def get_models(ctx):
    # Create the RNN and load its parameter settings
    rnn = mdn_rnn(z_dim, h_dim)  # z_dim and h_dim are import from settings.py
    # rnn.load_parameters(path_to_rnn_params, ctx=ctx)

    # Create the VAE and load its parameter settings
    vae = ConvVae()
    # vae.load_parameters(path_to_vae_params, ctx=ctx)

    # Create the controller and load its parameter settings
    controller = Controller()
    # controller.load_parameters(path_to_ctrl_params, ctx=ctx)

    return rnn, vae, controller


class RolloutGenerator:

    def __init__(self, world_model, agent, environment):
        self.world_model = world_model
        self.environment = environment
        self.agent = agent

    def rollout(self, controller):
        """
        environment, rnn, vae are global variables
        :param controller:
        :return: cumulative_reward
        """
        environment, vae, rnn = self.environment, self.vae, self.rnn

        end, reward, state = environment.reset()
        print(f'Initial - end: {end}, reward: {reward}, len state: {len(state)}')

        h = np.zeros(h_dim)  # h = rnn.reset_state()
        cumulative_reward = 0

        while (end == 0):
            # z = vae.encode(state)
            z = np.ones(z_dim)
            a = controller.action(z, h)
            end, reward, state = environment.step(a)
            cumulative_reward += reward
            # h = rnn.forward([a, z, h])

        return cumulative_reward


if __name__ == '__main__':
    ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()

    agent, environment = set_up_env()
    rnn, vae, controller = get_models(ctx)
    generator = RolloutGenerator(environment, rnn, vae)
    cumulative_reward = generator.rollout(controller)
    print(cumulative_reward)
    es_trainer = ES_trainer(generator, 100, 10)
    w, reward = es_trainer.train(n_iter=20)
    print(w, reward)