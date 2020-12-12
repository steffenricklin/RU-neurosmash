from mxnet import nd
from mxnet.gluon import nn

class LSTM(nn.HybridBlock):
    def __init__(self, x_dim, h_dim=5):
        super(LSTM, self).__init__()        
        """
        Initialization LSTM model.

        The parameters in this class are coded according to the scheme in RNN/lstm_coded.png and RNN/lstm_formulas.png

        :param z_dim (int) Input dimensions, latent vector of VAE  
        :param out_dim (int) Output dimension
        :param c_dim (int) Dimension of the memory

        return: LSTM object
        """

        # Initialize parameters
        self.x_dim = x_dim # Input
        self.h_dim = h_dim # Output dim
        self.c_dim = h_dim
        self.f_dim = h_dim
        self.o_dim = h_dim
        self.i_dim = h_dim

        # Weight initialization 
        # Only the output dim is specified, the input dim will be inferred at first use
        # Since the outputs of these matrices will always be summed, only one of every (W,U) needs a bias term
        with self.name_scope():

            self.W_f = nn.Dense(self.f_dim, in_units = self.x_dim, use_bias=True)
            self.U_f = nn.Dense(self.f_dim, in_units = self.h_dim, use_bias=False)

            self.W_i = nn.Dense(self.i_dim, in_units = self.x_dim,use_bias=True)
            self.U_i = nn.Dense(self.i_dim, in_units = self.h_dim,use_bias=False)
            
            self.W_o = nn.Dense(self.o_dim, in_units = self.x_dim,use_bias=True)
            self.U_o = nn.Dense(self.o_dim, in_units = self.h_dim,use_bias=False)

            self.W_c = nn.Dense(self.c_dim, in_units = self.x_dim,use_bias=True)
            self.U_c = nn.Dense(self.c_dim, in_units = self.h_dim,use_bias=False)

    def hybrid_forward(self, F, X, *args):
        """
        This method closely follows the formulas in RNN/lstm_formulas.png
        """
        h, c = args[0], args[1]
        f_t             = (self.W_f(X) + self.U_f(h)).sigmoid()
        i_t             = (self.W_i(X) + self.U_i(h)).sigmoid()
        o_t             = (self.W_o(X) + self.U_o(h)).sigmoid()
        c_tilde_t       = (self.W_c(X) + self.U_c(h)).sigmoid()
        new_c           = nd.multiply(f_t, c) + nd.multiply(i_t , c_tilde_t)
        new_h           = nd.multiply(o_t, c.sigmoid())
        return new_h, new_c

