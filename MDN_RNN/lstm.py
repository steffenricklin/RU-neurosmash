from mxnet import nd
from mxnet.gluon import nn

class LSTM(nn.Block):
    def __init__(self, x_dim, h_dim, c_dim = 5, i_dim=5, f_dim = 5, o_dim = 5):
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
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.o_dim = o_dim
        self.i_dim = i_dim
        
        # Surrogate output for first use
        self.h_0 = nd.zeros(self.h_dim)
        self.h = self.h_0

        # Surrogate memory for first use
        self.c_0 = nd.zeros(self.c_dim)
        self.c = self.c_0

        # Weight initialization 
        # Only the output dim is specified, the input dim will be inferred at first use
        # Since the outputs of these matrices will always be summed, only one of every (W,U) needs a bias term
        with self.name_scope():

            self.W_f = nn.Dense(self.f_dim, use_bias=True)
            self.U_f = nn.Dense(self.f_dim, use_bias=False)

            self.W_i = nn.Dense(self.i_dim, use_bias=True)
            self.U_i = nn.Dense(self.i_dim, use_bias=False)
            
            self.W_o = nn.Dense(self.o_dim, use_bias=True)
            self.U_o = nn.Dense(self.o_dim, use_bias=False)

            self.W_c = nn.Dense(self.c_dim, use_bias=True)
            self.U_c = nn.Dense(self.c_dim, use_bias=False)


    def forward(self, X):
        """
        This method closely follows the formulas in RNN/lstm_formulas.png
        """
        f_t = (self.W_f@X + self.U_f@self.h).sigmoid()
        i_t = (self.W_i@X + self.U_i@self.h).sigmoid()
        o_t = (self.W_o@X + self.U_o@self.h).sigmoid()
        c_tilde_t = (self.W_c@X + self.U_c@self.h).sigmoid()
        self.c = nd.multiply(f_t, self.c) + nd.multiply(i_t , c_tilde_t)
        self.h = nd.multiply(o_t, self.c.sigmoid())
        return self.h, self.c

    def reset_state(self):
        """
        This method resets the state to s_0, so that a new sequence can be started. 
        """
        self.h = self.h_0
        self.c = self.c_0

