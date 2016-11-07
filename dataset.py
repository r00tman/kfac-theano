import os

import numpy as np
import theano

from theano_utils import floatX

os.environ['KERAS_BACKEND'] = 'theano'
from keras.datasets import mnist

from sklearn.datasets import load_digits


def load_data(source='mnist', size=1200, downscale=2):
    if source == 'digits':
        X_tr, y_tr = load_digits(return_X_y=True)
    else:
        (X_tr, y_tr), (X_te, y_te) = mnist.load_data()

    def preprocess_dataset(X, y):
        if source == 'mnist':
            X = (floatX(X)/255)[:,::downscale,::downscale].reshape(-1, 28*28//(downscale**2))
        else:
            X = (floatX(X)/16).reshape(-1, 8, 8)[:,::downscale,::downscale].reshape(-1, 64//(downscale**2))

        outc = floatX(np.zeros((len(y), 10)))

        for i in range(len(y)):
            outc[i, y[i]] = 1.

        X, y, outc = X[:size], y[:size], outc[:size]

        X = theano.shared(X)
        y = theano.shared(y.astype('int32'))
        outc = theano.shared(outc)

        return X, y, outc

    X_tr, y_tr, outc_tr = preprocess_dataset(X_tr, y_tr)
    # X_te, y_te, outc_te = preprocess_dataset(X_te, y_te)

    return X_tr, y_tr, outc_tr
