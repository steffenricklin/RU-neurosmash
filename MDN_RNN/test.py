from mxnet import nd 
import mxnet as mx
from settings import *
from MDN_RNN.mdn_rnn import *
from MDN_RNN.MDN_RNN_trainer import MDN_RNN_trainer
n_samples = 1000


data_from_VAE = nd.random_normal(loc =0, scale = 1, shape = (n_samples,z_dim))
data_from_VAE= nd.expand_dims(data_from_VAE,0)
print(data_from_VAE.shape)

moves = nd.random_randint(low =0, high = 2, shape = n_samples)
move_one_hot = nd.array([[0,1] if m == 1 else [1,0] for m in moves])
move_one_hot = nd.expand_dims(move_one_hot,0)

print(data_from_VAE)
print(move_one_hot)

model = mdn_rnn(z_dim = z_dim + move_dim)
model.initialize("xavier")

trainer = MDN_RNN_trainer(0.001,1,5)

trainer.train(model,(data_from_VAE,move_one_hot) )

