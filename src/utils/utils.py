import mxnet as mx
import Neurosmash
from controller.Controller import Controller


def get_ctx(args):
    if args.device == "gpu" and mx.context.num_gpus():
        ctx = mx.gpu()
        print("Using device: GPU: {}".format(ctx))
        return ctx
    else:
        ctx = mx.cpu()
        if args.device == "gpu":
            print("No GPU found! ", end="")
            args.device = "cpu"
        print("Using device: CPU: {}".format(ctx))
        return ctx

def set_up_env(args):
    # Initialize agent and environment

    controller = Neurosmash.Agent()  # This is an example agent.
    if args.use_controller:
        controller = Controller()

    # This is the main environment.
    try:
        environment = Neurosmash.Environment(args)
    except:
        print("Connecting to environment failed. Please make sure Neurosmash is running and check your settings.")
        print(f"Settings from world model: ip={args.ip}, port={args.port}, size={args.size}, timescale={args.timescale}")
    else:
        print("Successfully connected to Neurosmash!")
        return controller, environment
