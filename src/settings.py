import os
os.getcwd()
project_dir = os.getcwd()
data_dir = f"{project_dir}/data"
param_dir = f"{data_dir}/parameters"
image_dir = f"{data_dir}/images"
path_to_rnn_params = f"{param_dir}/mdn_rnn.params"
path_to_vae_params = f"{param_dir}/vae.params"
path_to_ctrl_params = f"{param_dir}/controller.params"

z_dim = 10
h_dim = 10
move_dim = 2
# RU-neurosmash environment settings
ip = "127.0.0.1"  # Ip address that the TCP/IP interface listens to (127.0.0.1 by default)
port = 13000  # Port number that the TCP/IP interface listens to (13000 by default)
size = 64  # This is the size of the texture that the environment is rendered.
timescale = 1  # This is the simulation speed of the environment.
# model parameter settings
