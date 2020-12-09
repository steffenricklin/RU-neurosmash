from mxnet import nd 
import mxnet as mx
import numpy as np
from settings import *
from MDN_RNN.mdn_rnn import *
from MDN_RNN.MDN_RNN_trainer import MDN_RNN_trainer
n_samples = 100
# from gluonts.mx.distribution.multivariate_gaussian import MultivariateGaussian
# t_dim = 10
# n_components = 2
# mu = nd.random.randn(n_components,t_dim)
# sigma = np.array([0.5,0.5])
# dist = MultivariateGaussian(mu[0], nd.linalg.potrf(sigma[0] * nd.eye(t_dim)))
# sample = nd.random.randn(t_dim)
# print(dist.log_prob(sample))

data_from_VAE = nd.random_normal(loc=0, scale=1, shape=(n_samples,z_dim))
data_from_VAE= nd.expand_dims(data_from_VAE,0)

moves = nd.random_randint(low=0, high=2, shape=n_samples)
move_one_hot = nd.array([[0,1] if m == 1 else [1,0] for m in moves])
move_one_hot = nd.expand_dims(move_one_hot,0)

model = mdn_rnn(input_dim = z_dim + move_dim, interface_dim=128, output_dim = z_dim)
model.initialize("xavier")

trainer = MDN_RNN_trainer(0.01,k1=1,k2=10)

trainer.train(model,(data_from_VAE,move_one_hot) ,n_epochs = 100, print_every = 1)

