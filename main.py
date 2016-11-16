import timeit
from collections import OrderedDict

import numpy as np
import numpy.linalg
import numpy.random as nprng

import theano
import theano.tensor as T
import theano.printing

from theano_utils import floatX
from dataset import load_data
from nnet import NNet
from natgradopt import NatGradOpt

# theano.config.floatX = "float32"
# print(theano.config.floatX)
# theano.config.warn_float64 = 'pdb'

# -- load data --
# X_tr, y_tr, outc_tr = load_data('mnist', 1200, 2)
X_tr, y_tr, outc_tr = load_data('digits', 1800, 1)

x = T.matrix('x')

# -- model def --
model = NNet(x, [X_tr.get_value().shape[-1], 15, 15, outc_tr.get_value().shape[-1]])
# model = NNet(x, [X_tr.get_value().shape[-1], outc_tr.get_value().shape[-1]])
opt = NatGradOpt(model)

f = open("test.csv", "w")

for it in range(100):
    stats = opt.step(X_tr.get_value(), y_tr.get_value(), outc_tr.get_value())
    print(it, 'opt stats: ', 'loss:', stats['loss'], 'delta_ll:', stats['delta_ll'])
    print(it, '    extra: ', stats)
    print(stats['loss'], stats['grad_mean'], stats['rho'], opt.c_lambd_inv, file=f)
    f.flush()

f.close()
