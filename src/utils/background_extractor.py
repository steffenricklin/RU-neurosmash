# Imports
import numpy as np
import scipy.stats as sp
import Neurosmash
import matplotlib.pyplot as plt
import pickle
from src.settings import *
from os import path
from settings import *

class Background_Extractor:

    def extract_and_save_background(self, n_frames = 100):
        target_shape = (size,size,3)
        buffer = np.zeros((n_frames,size,size,3))
        agent = Neurosmash.Agent()
        environment = Neurosmash.Environment()
        frames_counter = 0
        end,reward,state = environment.reset()
        while frames_counter < n_frames:
            end = 0
            while end == 0:
                action = agent.step(end, reward, state)
                end, reward, state = environment.step(action)
                buffer[frames_counter] = np.reshape(state, target_shape)
                frames_counter+=1
                if frames_counter >= n_frames:
                    break
        mode, count = sp.mode(buffer, axis = 0)
        fig, ax = plt.subplots()
        ax.imshow(np.squeeze(mode)/255)
        plt.savefig(f"{image_dir}/background_{size}*{size}.png")
        plt.show()
        pickle.dump(np.squeeze(mode),open(f"{image_dir}/background_{size}*{size}.p","wb"))

    def get_background(self):
        target_filename = f"{image_dir}/background_{size}*{size}.p"
        if not path.exists(target_filename):
            self.extract_and_save_background()
        return pickle.load(open(target_filename,"rb"))

