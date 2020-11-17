import numpy as np
import Neurosmash
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn
ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()


class ConvVae(gluon.HybridBlock):
    
    def __init__(self, mu, sigma, size=64):
        """
        size: 
        mu:
        sigma:
        """
        self.size = size
        self.mu = mu
        self.sigma = sigma
        super(ConvVae, self).__init__()
        with self.name_scope():
            # define encoder
            self.encoder = self.getEncoder()

            # define decoder
            self.decoder = self.getDecoder()
        
    
    def state2image(self, state):
        return np.array(state, "uint8").reshape(self.size, self.size, 3)

    def getEncoder(self):
        # Input has the shape size x size x 3
        encoder = nn.HybridSequential(prefix='encoder')
        # relu conv 32 x 4
        encoder.add(nn.Conv2D(channels=32*4, strides=2, activation='relu'))
        # relu conv 64 x 4
        encoder.add(nn.Conv2D(channels=64*4, strides=2, activation='relu'))
        # relu conv 128 x 4
        encoder.add(nn.Conv2D(channels=128*4, strides=2, activation='relu'))
        # relu conv 256 x 4
        encoder.add(nn.Conv2D(channels=256*4, strides=2, activation='relu'))

        return encoder
        
    def get_vae(self, F, h):
        # Get mu and log-variance
        self.mu, self.lv = F.split(h, axis=1, num_outputs=2)
        # Calculate epsilon
        eps = F.random_normal(loc=0, scale=1, shape=(self.batch_size, self.n_latent), ctx=model_ctx)
        # Calculate z
        z = mu + F.exp(0.5*lv) * eps
        return z

    def getDecoder(self):
        # shape of z should be already reshape from (1024x1x1)  to 1x1x1024
        decoder = nn.HybridSequential(prefix='decoder')
        # relu deconv 128x5 to 5x5x128
        decoder.add(nn.Conv2DTranspose(channels=128*5, strides=2, activation='relu'))
        # relu deconv 64x5 to 13x13x64
        decoder.add(nn.Conv2DTranspose(channels=64*5, strides=2, activation='relu'))
        # relu deconv 32x6 to 30x30x32
        decoder.add(nn.Conv2DTranspose(channels=32*6, strides=2, activation='relu'))
        # sigmoid deconv 3x6 to 64x64x3
        decoder.add(nn.Conv2DTranspose(channels=128*5, strides=2, activation='sigmoid'))

        return decoder

    def hybrid_forward(self, F, x, train=False, *args, **kwargs):
        h = self.encoder(x)

        # generate z from h
        z = self.get_vae(F, h)

        if not train:
            return z  # return z
        else:  # reconstruct
            z = z.reshape((1, 1, 1024))
            reconstruction = self.decoder(z)

            self.output = reconstruction

            mu = self.mu
            lv = self.lv
            # Kullback-Leibler
            KL = 0.5 * F.sum(1 + lv - mu * mu - F.exp(lv), axis=1)

            logloss = F.sum(x*F.log(y+self.soft_zero)+ (1-x)*F.log(1-y+self.soft_zero), axis=1)
            loss = -logloss-KL
            return loss
