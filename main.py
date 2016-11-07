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
print(theano.config.floatX)
theano.config.warn_float64 = 'pdb'

# -- load data --
X_tr, y_tr, outc_tr = load_data('mnist', 1200, 2)

x = T.matrix('x')

# -- model def --
model = NNet(x, [X_tr.get_value().shape[-1], 10, outc_tr.get_value().shape[-1]])
opt = NatGradOpt(model)

for it in range(200):
    rho, t_r, delta_ll = opt.step(X_tr.get_value(), y_tr.get_value(), outc_tr.get_value())
    print('fuck theano', round(opt.c_lambd_inv, 9), round(opt.c_lambd, 9), round(rho, 2), delta_ll, t_r[0], t_r[1:])
