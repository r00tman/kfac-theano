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
from gdopt import GDOptimizer
from naivesoopt import NaiveSecondOrderOptimizer

import scipy.linalg

# theano.config.floatX = "float32"
print(theano.config.floatX)
# theano.config.warn_float64 = 'pdb'

# -- load data --
# dataset = ('mnist', 1200, 2, 'classify')
dataset = ('digits', 1500, 2, 'classify')
X_tr, y_tr, outc_tr = load_data(*dataset, data_type='train')
X_te, y_te, outc_te = load_data('digits', 300, 2, 'classify', data_type='test')

x = T.matrix('x')

# -- model def --
# layer_sizes = [X_tr.shape[-1], 15, 15, outc_tr.shape[-1]]
layer_sizes = [X_tr.shape[-1], 15, 15, 15, 15, outc_tr.shape[-1]]
# layer_sizes = [X_tr.shape[-1], 8, 5, 8, outc_tr.shape[-1]]
# layer_sizes = [X_tr.shape[-1], 8, 6, 4, 6, 8, outc_tr.shape[-1]]
# layer_sizes = [X_tr.shape[-1], 8, 6, 6, 5, 4, 5, 6, 6, 8, outc_tr.shape[-1]]
model = NNet(x, layer_sizes)
opt = NaiveSecondOrderOptimizer(model, 'kr_diag')
# opt = GDOptimizer(model, 'adam')
# opt = GDOptimizer(model, 'gd')

def format_test_name(dataset, opt):
    file_name = 'test'
    file_name += '_%s_%d_div_%d_%s' % dataset
    file_name += '_%s' % '-'.join([str(p) for p in opt.model.layer_sizes])
    file_name += '_%s' % opt.algo
    file_name += '_os_%d' % opt.over_sampling if opt.algo == 'fisher' or opt.algo.startswith('kr') else ''
    file_name += '_r_%g' % opt.rate
    if 'c_lambda_inv' in opt.__dict__:
        file_name += '_l_%.2e' % opt.c_lambd_inv
    file_name += '.csv'

    return file_name

file_name = format_test_name(dataset, opt)
f = open('tests/' + file_name, "w")
print('writing to', file_name)

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

for it in range(2000):
    batch = nprng.permutation(len(X_tr))[:]
    stats = opt.step(X_tr[batch], y_tr[batch], outc_tr[batch])
    print(it, 'opt stats: ', 'loss:', stats['loss'], 'delta_ll:', stats['delta_ll'])
    print(it, '    extra: ', *stats.items())

    # plt.imshow(np.log10(stats['F']))
    # plt.colorbar()
    # plt.subplot(122)
    # plt.imshow(np.log10(stats['Fr']))
    # plt.colorbar()
    # plt.show()
    # print(scipy.linalg.norm(stats['F']), scipy.linalg.norm(stats['Fr']), scipy.linalg.norm(stats['F']-stats['Fr']))
    if it == 0:
        print(*stats.keys(), sep=',', file=f)
    print(*stats.values(), sep=',', file=f)
    f.flush()

f.close()
