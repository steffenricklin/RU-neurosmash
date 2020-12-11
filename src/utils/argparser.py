import argparse
from settings import *
from distutils.util import strtobool


def get_args():
    # collecting settings from arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="world_model")
    parser.add_argument("--save_model", default="world_model")
    parser.add_argument("--load_model", default=True, type=lambda x: bool(strtobool(str(x))))
    parser.add_argument("--vision_model", default="classifier", help="vae | classifier")
    parser.add_argument("--vae_remove_background", default=True, type=lambda x: bool(strtobool(str(x))),
                        help="If vision_model is vae, the inputs background will be removed")
    parser.add_argument("--use_controller", default=True, type=lambda x: bool(strtobool(str(x))))
    parser.add_argument("--continue_training", default=False, type=lambda x: bool(strtobool(str(x))),
                        help="Whether to train at all. Default is False to prevent overwriting well-performing modules")
    parser.add_argument("--train_vision", default=True, type=lambda x: bool(strtobool(str(x))))
    parser.add_argument("--train_rnn", default=True, type=lambda x: bool(strtobool(str(x))))
    parser.add_argument("--train_ctrl", default=True, type=lambda x: bool(strtobool(str(x))))

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

    parser.add_argument("--device", default="gpu", help="cpu | gpu")
    parser.add_argument("--batch_size", default=40)
    parser.add_argument("--learning_rate", default=1e-3)
    parser.add_argument("--n_epochs", default=15)
    parser.add_argument("--train_split", default=0.9)

    # parameter paths
    parser.add_argument("--path_to_clf_params", default=path_to_clf_params)
    parser.add_argument("--path_to_rnn_params", default=path_to_rnn_params)
    parser.add_argument("--path_to_vae_params", default=path_to_vae_params)
    parser.add_argument("--path_to_ctrl_params", default=path_to_ctrl_params)

    # rollout simulations
    parser.add_argument("--rounds", default=5, type=lambda x: int(x),
                        help="number of rounds simulated with rollout")

    args = parser.parse_args()

    print("Load parameters?:", args.load_model)
    print("Use controller?:", args.use_controller, "(False=random agent)")

    return args
