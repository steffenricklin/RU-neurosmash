import numpy as np
import Neurosmash
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn


class Reshape(gluon.HybridBlock):
    """Source: https://github.com/dingran/vae-mxnet/blob/master/code/vaecnn-gluon.ipynb
    """
    def __init__(self, target_shape, **kwargs):
        super().__init__(**kwargs)
        self.target_shape = target_shape

    def hybrid_forward(self, F, x):
        #print(x.shape)
        return x.reshape((0, *self.target_shape)) # setting the first axis to 0 to copy over the original shape, i.e. batch_size

    def __repr__(self):
        return self.__class__.__name__

    
class Expand_dims(gluon.HybridBlock):
    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def hybrid_forward(self, F, x):
        #print(x.shape)
        return x.expand_dims(axis=self.axis)

    def __repr__(self):
        return self.__class__.__name__
    

class ConvVae(gluon.HybridBlock):
    """
    https://github.com/hardmaru/WorldModelsExperiments/blob/master/carracing/vae/vae.py
    """
    def __init__(self, batch_size=100, size=64, z_size=32, KL_tolerance_value= 0.5, **kwargs):
        """
        size:
        """
        self.soft_zero = 1e-10
        self.output = None
        self.mu = None
        self.batch_size = batch_size
        self.n_latent = 2
        self.size = size
        self.z_size = z_size
        self.KL_tolerance_value = KL_tolerance_value
        self.ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
        super(ConvVae, self).__init__(**kwargs)
        with self.name_scope():
            # define encoder
            self.encoder = self.getEncoder()
            
            # define decoder
            self.decoder = self.getDecoder()
        
    
    def state2image(self, state):
        return np.array(state, "uint8").reshape(self.size, self.size, 3)

    def getEncoder(self):
        # Input has the shape size x size x 3
        encoder = nn.HybridSequential()
        # relu conv 32 x 4
        encoder.add(nn.Conv2D(channels=32, kernel_size=4, strides=2, activation='relu'))
        # relu conv 64 x 4
        encoder.add(nn.Conv2D(channels=64, kernel_size=4, strides=2, activation='relu'))
        # relu conv 128 x 4
        encoder.add(nn.Conv2D(channels=128, kernel_size=4, strides=2, activation='relu'))
        # relu conv 256 x 4
        encoder.add(nn.Conv2D(channels=256, kernel_size=4, strides=2, activation='relu'))
        # dense layer for mu and log-var
        encoder.add(nn.Flatten())
        self.dense_mu = nn.Dense(units=self.z_size)
        self.dense_lv = nn.Dense(units=self.z_size)
        return encoder
        
    def get_vae(self, F, h):
        # Get mu and log-variance
        self.mu = self.dense_mu(h)
        self.lv = self.dense_lv(h)
        # self.mu, self.lv = F.split(h, axis=1, num_outputs=2)

        # Calculate epsilon
        eps = F.random_normal(loc=0, scale=1, shape=(self.batch_size, self.z_size), ctx=self.ctx)
        # Calculate z
        self.sigma = F.exp(0.5 * self.lv)
#         z = F.broadcast_add(self.mu, F.broadcast_mul(z, eps))
        z = self.mu + self.sigma * eps
        return z

    def getDecoder(self):
        decoder = nn.HybridSequential(prefix='decoder')
        decoder.add(nn.Dense(units=1024))
        decoder.add(Expand_dims(axis=-1))
        decoder.add(Expand_dims(axis=-1))
        # relu deconv 128x5 to 5x5x128
        decoder.add(nn.Conv2DTranspose(channels=128, kernel_size=5, strides=2, activation='relu'))
        # relu deconv 64x5 to 13x13x64
        decoder.add(nn.Conv2DTranspose(channels=64, kernel_size=5, strides=2, activation='relu'))
        # relu deconv 32x6 to 30x30x32
        decoder.add(nn.Conv2DTranspose(channels=32, kernel_size=6, strides=2, activation='relu'))
        # sigmoid deconv 3x6 to 64x64x3
        decoder.add(nn.Conv2DTranspose(channels=3, kernel_size=6, strides=2, activation='sigmoid'))

        return decoder

    def hybrid_forward(self, F, x, *args, **kwargs):
        h = self.encoder(x)

        # generate z from h
        z = self.get_vae(F, h)
        
        y = self.decoder(z)
        self.output = y
        
        # Mu and log-variance
        mu = self.mu
        lv = self.lv

        # reconstruction loss
        self.r_loss = F.sum(F.square(x - y), axis=[1,2,3])
        self.r_loss = F.mean(self.r_loss)

        self.KL = -0.5 * F.sum(1 + lv - F.square(mu) - F.exp(lv), axis=1)
        self.KL = F.maximum(self.KL, self.KL_tolerance_value * self.z_size)
        self.KL = F.mean(self.KL)

        loss = self.r_loss + self.KL

        # Kullback-Leibler
#        KL = 0.5 * F.sum(1 + lv - mu * mu - F.exp(lv), axis=1)
#
#        xlog = x*F.log(y+self.soft_zero)
#        xminlog = (1-x)*F.log(1-y+self.soft_zero)
#         logloss = F.sum(xlog + xminlog, axis=[1,2,3])
#        logloss = F.sum(xlog + xminlog, axis=1)
#        loss = -logloss-KL
        return self.output, loss
