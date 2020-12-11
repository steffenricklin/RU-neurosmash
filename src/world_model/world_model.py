import mxnet as mx
from classifier.agent_location_classifier import Agent_Location_Classifier
from vae.convvae import ConvVae

class World_Model:
    """Combines a vision model (vae or classifier),
    a mdn_rnn module and a controller module.
    """
    def __init__(self, args):

        # if args.vision_model == "classifier":
        #     self.vision = Agent_Location_Classifier()
        # else:
        #     self.vision = ConvVae()
        self.vision = Agent_Location_Classifier if args.vision_model == "classifier" else ConvVae()
        rnn = None
        controller = None

    def train(self):
        pass

    def test(self):
        pass

    def rollout(self):
