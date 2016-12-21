import numpy as np
import numpy.random as nprng

import theano
import theano.tensor as T

from theano_utils import floatX

class LogReg:
    def __init__(self, inp, shape, act=T.nnet.sigmoid):
        self.shape = shape
        print(shape)

        self.W = theano.shared(
            value=floatX(nprng.randn(shape[0], shape[1])*np.sqrt(2/shape[1])),
            # value=floatX(nprng.randn(shape[0], shape[1])*np.sqrt(2/(shape[1] + shape[0]))),
            name='W',
            borrow=True
        )

        # self.b = theano.shared(
        #     value=floatX(nprng.randn(shape[0])*np.sqrt(2/shape[0])),
        #     name='b',
        #     borrow=True
        # )

        # self.s = T.dot(self.W, inp.T).T + self.b
        self.s = T.dot(self.W, inp.T).T
        self.a = act(self.s)

        # self.params = [self.W, self.b]
        self.params = [self.W]

        self.inp = inp
