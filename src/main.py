from utils.utils import *
from utils.argparser import get_args
from world_model.world_model import World_Model


def run():
    # Connect to Neurosmash
    controller, environment = set_up_env(args)

    # Create and load the world_model
    args.ctx = get_ctx(args)
    world_model = World_Model(controller, environment, args)

    world_model.train(args)  # trains some of the modules if set so by args

    # simulate rounds in Neurosmash without training
    cum_r = world_model.rollout(controller, args.rounds)
    print("cumulative reward:", cum_r)


if __name__ == '__main__':
    # TODO everyone: update if more parts work!: currently recommended arguments: --load_model False --timescale 5 --rounds 10 --use_controller False
    args = get_args()
    run()
