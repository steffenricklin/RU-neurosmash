import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn
from sklearn.model_selection import train_test_split
# !pip install tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm import tqdm
from matplotlib import pyplot as plt

# for Neurosmash
import PIL
import Neurosmash

# our settings and models
from settings import *
from MDN_RNN.mdn_rnn import mdn_rnn
from vae.convvae import ConvVae


def set_up_env():
    # Initialize agent and environment

    # This is an example agent.
    agent = Neurosmash.Agent()

    # This is the main environment.
    try:
        environment = Neurosmash.Environment(ip, port, size, timescale)
    except:
        print("Connecting to environment failed. Please make sure Neurosmash is running and check your settings.")
    else:
        print("Connection to environment established successfully!")
    return agent, environment

def get_models(ctx):

    # Create the RNN and load its parameter settings
    rnn = mdn_rnn(z_dim, h_dim)  # z_dim and h_dim are import from settings.py
    rnn.load_parameters(path_to_rnn_params, ctx=ctx)

    # Create the VAE and load its parameter settings
    vae = ConvVae()
    vae.load_parameters(path_to_vae_params, ctx=ctx)

    # Create the controller and load its parameter settings
    controller = None
    # controller.load_parameters(path_to_ctrl_params, ctx=ctx)
    
    return rnn, vae, controller


def rollout(controller):
    """
    environment, rnn, vae are global variables
    :param controller:
    :return: cumulative_reward
    """
    obs = environment.reset()
    h = rnn.reset_state()
    done = False
    cumulative_reward = 0
    while not done:
        z = vae.encode(obs)
        a = controller.action(z, h)
        obs, reward, done = environment.step(a)
        cumulative_reward += reward
        h = rnn.forward([a, z, h])
    return cumulative_reward


if __name__ == '__main__':
    ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()
    
    agent, environment = set_up_env()
    rnn, vae, controller = get_models(ctx)
    
    cum_reward = rollout(controller)
