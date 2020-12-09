from settings import *
import numpy as np
from utils.converters import *
from mxnet import nd

def locate_agents(state):
    """
    Receives a 1-d state
    :param state:
    :return:
    """
    state = state.asnumpy()
    redloc = np.unravel_index(np.array(state[0::3]).argmax(),(size,size))
    blueloc = np.unravel_index(np.array(state[2::3]).argmax(),(size,size))
    locations = nd.array([*redloc, *blueloc]).astype(np.float32)/size
    return locations



