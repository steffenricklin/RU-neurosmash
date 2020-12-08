z_dim = 10
h_dim = 10
move_dim = 2
# RU-neurosmash environment settings
ip = "127.0.0.1"  # Ip address that the TCP/IP interface listens to (127.0.0.1 by default)
port = 13000  # Port number that the TCP/IP interface listens to (13000 by default)
size = 64  # This is the size of the texture that the environment is rendered.
timescale = 5  # This is the simulation speed of the environment.
# model parameter settings
path_to_rnn_params = "parameters/mdn_rnn.params"
path_to_vae_params = "parameters/vae.params"
path_to_ctrl_params = "parameters/controller.params"

