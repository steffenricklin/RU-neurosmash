# Imports
import numpy as np
import scipy.stats as sp
import Neurosmash
import matplotlib.pyplot as plt
import pickle
from src.settings import *
from os import path
from settings import *
from mxnet import nd
class Background_Extractor:

    def __init__(self, env, agent, args):
        self.args=  args
        self.env = env
        self.agent = agent

    def extract_and_save_background(self, n_frames = 100):
        target_shape = (self.args.size,self.args.size,3)
        buffer = np.zeros((n_frames,self.args.size,self.args.size,3))
        frames_counter = 0
        end,reward,state = self.env.reset()
        while frames_counter < n_frames:
            end = 0
            while end == 0:
                action = self.agent.step(end, reward, state)
                end, reward, state = self.env.step(action)
                buffer[frames_counter] = np.reshape(state, target_shape)
                frames_counter+=1
                if frames_counter >= n_frames:
                    break
        mode, count = sp.mode(buffer, axis = 0)
        fig, ax = plt.subplots()
        ax.imshow(np.squeeze(mode)/255)
        plt.savefig(f"{image_dir}/background_{self.args.size}*{self.args.size}.png")
        plt.show()
        pickle.dump(np.squeeze(mode),open(f"{image_dir}/background_{self.args.size}*{self.args.size}.p","wb"))

    def get_background(self, oned = False):
        target_filename = f"{image_dir}/background_{self.args.size}*{self.args.size}.p"
        if not path.exists(target_filename):
            self.extract_and_save_background()
        if oned:
            return np.reshape(pickle.load(open(target_filename,"rb")),-1)
        else:
            return pickle.load(open(target_filename,"rb"))

    def clean_and_reshape(self, state, oned=False):
        """
        Receives a 1-d list with pixel values, and returns an mx tensor reshaped as an image, where all the pixels that are background are set to 0
        :param state:
        :param oned:
        :return:
        """
        bg = self.get_background(oned = False)
        state_rs = np.reshape(state, (self.args.size, self.args.size, 3))
        def func(x):
            redequal = x[:, :, 0] == bg[:, :, 0]
            greenequal = x[:, :, 1] ==bg[:, :, 1]
            blueequal = x[:, :, 2] == bg[:, :, 2]
            return np.repeat((redequal & greenequal & blueequal)[:,:,None], 3,2)
        im = np.where(func(state_rs),0, state_rs)
        flat = np.reshape(im,(-1,))
        if oned:
            return nd.array(flat)
        else:
            tensor = np.reshape(flat, (1, 3,self.args.size, self.args.size))
            return nd.array(tensor)

