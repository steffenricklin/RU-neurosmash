# Imports
import numpy as np
import scipy.stats as sp
import Neurosmash
import matplotlib.pyplot as plt
import pickle
from settings import *

import os.path
from os import path

class Background_Extractor:
    def __init__(self, ip = "127.0.0.1", port = 13000, timescale = 1):
        print(project_dir)
        self.ip =ip
        self.port =port
        self.timescale = timescale
        self.loc = f"{project_dir}/background_images"

    def extract_and_save_background(self, size, n_frames = 100):
        target_shape = (size,size,3)
        buffer = np.zeros((n_frames,size,size,3))
        agent = Neurosmash.Agent()
        environment = Neurosmash.Environment(self.ip, self.port, size, self.timescale)
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
        plt.savefig(f"{self.loc}/background_{size}*{size}.png")
        plt.show()
        pickle.dump(np.squeeze(mode),open(f"{self.loc}/background_{size}*{size}.p","wb"))

    def get_background(self, size):
        target_filename = f"{self.loc}/background_{size}*{size}.p"
        if not path.exists(target_filename):
            self.extract_and_save_background(size)
        return pickle.load(open(target_filename,"rb"))







extr = Background_Extractor()
extr.get_background(128)
#
# background = pickle.load(open(f"{loc}/background_{size}*{size}.p","rb"))
#
# n_test_frames = 10
# frames_counter = 0
# while games_counter < n_games:
#     end = 0
#     while end == 0:
#         action = agent.step(end, reward, state)
#         end, reward, state = environment.step(action)
#         state_rs = np.reshape(state, target_shape)
#         clean_state = np.where(state_rs == background, 100, state_rs)
#         fig,ax = plt.subplots(1,2)
#         ax[0].imshow(state_rs)
#         ax[1].imshow(clean_state)
#         plt.show()
#         frames_counter+=1
#         if frames_counter >= n_test_frames:
#             break
#     if frames_counter >= n_test_frames:
#         break
