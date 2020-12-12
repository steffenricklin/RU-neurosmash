from mxnet import nd, autograd,gluon
import mxnet.gluon.nn as nn
import utils.background_extractor as BG
class Agent_Location_Classifier(gluon.Block):

    def __init__(self):
        super(Agent_Location_Classifier, self).__init__()
        with self.name_scope():
            self.conv1 = nn.Conv2D(channels=6 , kernel_size = 5, strides =  2,use_bias=True)
            self.conv2 = nn.Conv2D(channels=12, kernel_size = 5, strides =  2,use_bias=True)
            self.conv3 = nn.Conv2D(channels=24, kernel_size = 5, strides =  2,use_bias=True)
            self.linear1 = nn.Dense(units = 200,activation = 'relu',use_bias=True)
            self.linear2 = nn.Dense(units = 100, activation='relu',use_bias=True)
            self.out = nn.Dense(units = 4, activation='sigmoid',use_bias=True)

    def forward(self,A):
        A = self.conv1(A)
        A = self.conv2(A)
        A = self.conv3(A)
        A = nd.reshape(A, (1,-1))
        A = self.linear1(A)
        A = self.linear2(A)
        A = self.out(A)
        return A



