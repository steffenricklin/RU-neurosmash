import numpy as np
import Neurosmash
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn


class ConvVae(gluon.HybridBlock):
    
    def __init__(self, batch_size=100, size=64, **kwargs):
        """
        size:
        """
        self.soft_zero = 1e-10
        self.output = None
        self.mu = None
        self.batch_size = batch_size
        self.n_latent = 2
        self.size = size
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
        encoder.add(nn.Conv2D(channels=32, in_channels=3, kernel_size=1, strides=2, activation='relu'))
        # relu conv 64 x 4
        encoder.add(nn.Conv2D(channels=64, in_channels=32, kernel_size=1, strides=2, activation='relu'))
        # relu conv 128 x 4
        encoder.add(nn.Conv2D(channels=128, in_channels=64, kernel_size=1, strides=2, activation='relu'))
        # relu conv 256 x 4
        encoder.add(nn.Conv2D(channels=256, in_channels=128, kernel_size=1, strides=2, activation='relu'))

        return encoder
        
    def get_vae(self, F, h):
        # Get mu and log-variance
        self.mu, self.lv = F.split(h, axis=1, num_outputs=2)
        # Calculate epsilon
        eps = F.random_normal(loc=0, scale=1, shape=(self.batch_size, self.n_latent), ctx=self.ctx)
        # Calculate z
        print(self.mu.shape)
        print(eps.shape)
        z = F.exp(0.5 * self.lv)
        z = F.broadcast_mul(z, eps)
        z = F.broadcast_add(self.mu, z)
        return z

    def getDecoder(self):
        # shape of z should be already reshape from (1024x1x1)  to 1x1x1024
        decoder = nn.HybridSequential(prefix='decoder')
        decoder.add(nn.Dense(units=4, in_units=256, activation='relu'))
        # relu deconv 128x5 to 5x5x128
        decoder.add(nn.Conv2DTranspose(channels=128, in_channels=1024, kernel_size=1, strides=2, activation='relu'))
        # relu deconv 64x5 to 13x13x64
        decoder.add(nn.Conv2DTranspose(channels=64, in_channels=128, kernel_size=1, strides=2, activation='relu'))
        # relu deconv 32x6 to 30x30x32
        decoder.add(nn.Conv2DTranspose(channels=32, in_channels=64, kernel_size=1, strides=2, activation='relu'))
        # sigmoid deconv 3x6 to 64x64x3
        decoder.add(nn.Conv2DTranspose(channels=3, in_channels=32, kernel_size=1, strides=2, activation='sigmoid'))

        return decoder

    def hybrid_forward(self, F, x, *args, **kwargs):
#         print("x.shape: ", x)
        h = self.encoder(x)
#         print("h.shape: ", h)

        # generate z from h
        z = self.get_vae(F, h)
        
#         print("z.shape: ", z)

#             z = z.reshape((1, 1, 1024))
        y = self.decoder(z)
        self.output = y

#         # Mu and log-variance
#         mu = self.mu
#         lv = self.lv

#         # Kullback-Leibler
#         KL = 0.5 * F.sum(1 + lv - mu * mu - F.exp(lv), axis=1)

#         logloss = F.sum(x*F.log(y+self.soft_zero)+ (1-x)*F.log(1-y+self.soft_zero), axis=1)
#         loss = -logloss-KL
#         return loss
        return y
