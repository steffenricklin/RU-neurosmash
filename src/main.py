from settings import *
from utils.rollout import set_up_env
import torch
import argparse
import mxnet as mx

from world_model.world_model import World_Model
from utils.utils import get_ctx
from utils.rollout import RolloutGenerator


def run():
    # Connect to Neurosmash
    agent, environment = set_up_env()

    # Create and load the world_model
    args.ctx = get_ctx(args)
    world_model = World_Model(args)
    if args.load_model:
        world_model = world_model.load(args)

    generator = RolloutGenerator(world_model, agent, environment)
    generator.rollout(args)

    # save world_model or not


if __name__ == '__main__':
    # collecting settings from arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="world_model")
    parser.add_argument("--save_model", default="world_model")
    parser.add_argument("--load_model", default=True)
    parser.add_argument("--vision_model", default="classifier", help="vae | classifier")
    parser.add_argument("--vae_remove_background", default=True,
                        help="If vision_model is vae, the inputs background will be removed")
    parser.add_argument("--train_vision", default=True)
    parser.add_argument("--train_rnn", default=True)
    parser.add_argument("--train_ctrl", default=True)

    parser.add_argument("--z_dim", default=z_dim)
    parser.add_argument("--h_dim", default=h_dim)
    parser.add_argument("--move_dim", default=move_dim)
    parser.add_argument("--ip", default=ip,
                        help="Ip address that the TCP/IP interface listens to (127.0.0.1 by default)")
    parser.add_argument("--port", default=port,
                        help="Port number that the TCP/IP interface listens to (13000 by default)")
    parser.add_argument("--size", default=size,
                        help="This is the size of the texture that the environment is rendered.")
    parser.add_argument("--timescale", default=timescale,
                        help="This is the simulation speed of the environment.")
    parser.add_argument("--state_dim", default=state_dim)

    # training
    parser.add_argument("--continue_training", default=True)
    parser.add_argument("--device", default="gpu", help="cpu | gpu")
    parser.add_argument("--batch_size", default=40)
    parser.add_argument("--learning_rate", default=1e-3)
    parser.add_argument("--n_epochs", default=15)
    parser.add_argument("--train_split", default=0.9)

    # rollout simulations
    parser.add_argument("--rounds", default=5, help="number of rounds simulated with rollout")

    args = parser.parse_args("")

    run()
