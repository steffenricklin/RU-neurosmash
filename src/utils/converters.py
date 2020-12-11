from settings import *
import numpy as np
import mxnet as mx
import classifier

def state2image(state):
    image = np.reshape(state, (size, size, 3))
    return image / 255

def state2tensor(state):
    inpt = np.reshape(state, (1, 3, size, size)).astype(np.uint8)
    inpt = np.interp(inpt, [0, 255], [0.0, 1.0])
    inpt = mx.nd.array(inpt)
    return inpt

def state2target(state):
    reshaped_state = np.reshape(state, (3, size, size))
    ndstate = mx.nd.array(reshaped_state)
    image = mx.nd.array(np.reshape(state, (1, 3, size, size)))
    target = classifier.agent_locator.locate_agents(state)/size
    return image, target