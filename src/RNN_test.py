import Neurosmash
from MDN_RNN.mdn_rnn import mdn_rnn
from classifier.agent_location_classifier import Agent_Location_Classifier
from classifier.agent_locator import locate_agents
from settings import *
from utils.argparser import get_args
from utils.background_extractor import Background_Extractor
from mxnet import nd
import numpy as np
import matplotlib.pyplot as plt


args = get_args()
vision = Agent_Location_Classifier()
vision.load_parameters(path_to_clf_params)

rnn = mdn_rnn(input_dim=7, interface_dim=10, output_dim=4)
rnn.load_parameters(path_to_rnn_params)

env = Neurosmash.Environment(args)
agent = Neurosmash.Agent()
end, reward, previous_state = env.reset()
n_steps = 30

extr = Background_Extractor(env, agent, args)
background = extr.get_background(oned=True)
h, c = (nd.zeros((1, rnn.RNN.h_dim)), nd.zeros((1, rnn.RNN.c_dim)))
eye = nd.eye(args.move_dim)
prev_pred = nd.zeros((1,4))
while end == 0:    # Get latent representation from LSTM
    z = vision(extr.clean_and_reshape(previous_state, args.size)/255)

    # Make random step
    a = np.random.randint(0,3)

    # Take the step and store the reward
    end, reward, state = env.step(a)

    # Get the new hidden states by feeding the RNN the previous state and the action
    action_onehot = eye[None, a]
    rnn_input = nd.concatenate([z, action_onehot], 1)
    pz, h, c = rnn(rnn_input, h, c)

    pred = pz.sample(1)

    one_d_without_background = np.where(state ==background, 10, state)
    # Plot actual position and predicted position
    [pxr, pyr, pxb, pyb] = map(lambda x: min(args.size-1,int(x*args.size)), list(prev_pred[0].asnumpy()))
    [xr, yr, xb, yb] = map(int,locate_agents(nd.array(one_d_without_background),args).asnumpy()*args.size)
    fix, ag = plt.subplots()
    picture = np.zeros((args.size, args.size, 3))
    picture[pxr, pyr, 0] = 0.5
    picture[pxb, pyb, 2] = 0.5

    picture[xr, yr,0] = 1
    picture[xb, yb,2] = 1
    plt.imshow(picture)
    plt.show()

    prev_pred = pred
    previous_state = state