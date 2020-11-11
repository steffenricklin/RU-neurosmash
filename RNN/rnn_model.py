from mxnet import nd            # This is for the array types. Use nd.array.  Similar to numpy by design 
                                # https://mxnet.apache.org/versions/1.7.0/api/python/docs/api/ndarray/index.html

from mxnet.gluon import nn      # This is for the neural network functionality
                                # https://mxnet.apache.org/versions/1.7.0/api/python/docs/api/gluon/index.html

class RNN_GMN(nn.Block):
    def __init__(self, a):
        self.a = a
        pass