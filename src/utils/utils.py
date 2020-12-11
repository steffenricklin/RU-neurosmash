import mxnet as mx


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
