import timeit
import os
from collections import OrderedDict

import numpy as np
import numpy.linalg
import numpy.random as nprng

import theano
import theano.gradient
import theano.tensor.slinalg
import theano.tensor as T
from theano.tensor.nlinalg import matrix_inverse

os.environ['KERAS_BACKEND'] = 'theano'
from keras.datasets import mnist

import theano.gof

theano.config.floatX = "float64"
print(theano.config.floatX)


class Rank(theano.gof.Op):
    """
    Matrix rank. Input should be a square matrix.
    """

    __props__ = ()

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        assert x.ndim == 2
        o = T.scalar(dtype=x.dtype)
        return theano.gof.Apply(self, [x], [o])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        try:
            z[0] = numpy.asarray(np.linalg.matrix_rank(x), dtype=x.dtype)
        except Exception:
            print('Failed to compute determinant', x)
            raise

    def grad(self, inputs, g_outputs):
        gz, = g_outputs
        x, = inputs
        raise NotImplementedError("lol, just why'd you ask this??")

    def infer_shape(self, node, shapes):
        return [()]

    def __str__(self):
        return "Rank"
rank = Rank()


def floatX(a):
    return np.asarray(a, dtype=theano.config.floatX)


def get_pred(act):
    return T.argmax(act, axis=1)


class LogReg:
    def __init__(self, inp, shape, act=T.nnet.sigmoid):
        self.shape = shape
        print(shape)

        self.W = theano.shared(
            value=floatX(nprng.randn(shape[0], shape[1])*np.sqrt(2/shape[1])),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            value=floatX(nprng.randn(shape[0])*np.sqrt(2/shape[0])),
            name='b',
            borrow=True
        )

        self.s = T.dot(self.W, inp.T).T + self.b
        self.a = act(self.s)

        self.pred = get_pred(self.a)
        self.params = [self.W, self.b]

        self.inp = inp


def get_loss_samples(act, target):
    return T.mean((act - target)**2/2, 1)


def get_loss(act, target):
    return T.mean(get_loss_samples(act, target))


def get_regularizer(params, w=1e-5):
    res = 0
    for p in params:
        res += T.sum(p**2)*w
    return res


def get_error(pred, target):
    return T.mean(T.neq(pred, target))


def get_total_loss(act, target, params, w):
    return get_loss(act, target) + get_regularizer(params, w)


# -- load data --
from sklearn.datasets import load_digits
X_tr, y_tr = load_digits(return_X_y=True)
# (X_tr, y_tr), (X_te, y_te) = mnist.load_data()

def preprocess_dataset(X, y):
    # X = (floatX(X)/255)[:,::7,::7].reshape(-1, 28*28//49)
    X = (floatX(X)/16).reshape(-1, 8, 8)[:,::1,::1].reshape(-1, 64//1)
    outc = floatX(np.zeros((len(y), 10)))

    for i in range(len(y)):
        outc[i, y[i]] = 1.

    # X, y, outc = X[:1000], y[:1000], outc[:1000]

    X = theano.shared(X)
    y = theano.shared(y.astype('int32'))
    outc = theano.shared(outc)

    return X, y, outc

X_tr, y_tr, outc_tr = preprocess_dataset(X_tr, y_tr)
# X_te, y_te, outc_te = preprocess_dataset(X_te, y_te)

# -- model def --
x = T.matrix('x')
y = T.ivector('y')
outc = T.matrix('outc')
rand_outc = T.matrix('rand_outc')

logreg = LogReg(x, (10, X_tr.get_value().shape[-1]))
logreg2 = LogReg(logreg.a, (outc_tr.get_value().shape[-1], 20))

lambd = T.scalar('lambd')
# -- target def --
from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(seed=234)

# loss_samples = get_loss_samples(logreg2.a, outc)
loss_samples = get_loss_samples(logreg.a, rand_outc*lambd)
loss = get_loss(logreg.a, outc)
err = get_error(get_pred(logreg.a), y)

params = logreg.params  # + logreg2.params

grad = []
grad2d = []
updates = OrderedDict()

print_pls = []

for p in params:
    grad += [T.grad(loss, p)]
    grad2d += [T.jacobian(loss_samples, p)]
    if grad2d[-1].ndim == 2:
        grad2d[-1] = grad2d[-1].dimshuffle(0, 1, 'x')

grad_vec = T.concatenate([g.flatten() for g in grad])
grad2d_vec = T.concatenate([g.flatten(2).T for g in grad2d]).T

# print_pls += [grad_vec.shape, grad2d_vec.shape]
# tensor wise: F_p,i,j = sum_k grad2d[p,i,k]*grad2d[p,k,j]
# F = T.batched_dot(grad2d_vec.dimshuffle(0, 1, 'x'), grad2d_vec.dimshuffle(0, 'x', 1))
# F = T.mean(F, 0)
F = T.dot(grad2d_vec.T, grad2d_vec)/grad2d_vec.shape[0]
print_pls += [F.dot(grad_vec).shape, F.shape, rank(F), T.mean(grad_vec**2)**0.5, (F**2).trace()]

new_grad_vec = theano.tensor.slinalg.solve(F+T.identity_like(F)*1e-6, grad_vec)
new_grad = []

offset = 0
for p in params:
    pval = p.get_value()
    new_grad += [new_grad_vec[offset:offset+pval.size].reshape(pval.shape)]
    offset += pval.size

    updates[p] = p - new_grad[-1]

# -- combining --
import theano.printing

outc_tr_shape = outc_tr.get_value().shape
rand_outc_tr = theano.shared(floatX(nprng.randn(*outc_tr_shape)*np.sqrt(2)))
# print_pls += [T.eq(T.sum(grad2d_vec, 0), 0)]

get_params = theano.function(
    inputs=[],
    outputs=params,
    on_unused_input='warn'
)

print_pls += [new_grad_vec.dot(F.dot(new_grad_vec))/2]

train = theano.function(
    inputs=[lambd],
    outputs=[new_grad_vec.dot(F.dot(new_grad_vec))/2+grad_vec.dot(new_grad_vec), loss, err] + print_pls,
    updates=updates,
    givens={
        x: X_tr,
        y: y_tr,
        outc: outc_tr,
        rand_outc: rand_outc_tr
    },
    on_unused_input='warn'
)

eva = theano.function(
    inputs=[],
    outputs=[loss],
    updates=[],
    givens={
        x: X_tr,
        y: y_tr,
        outc: outc_tr
    },
    on_unused_input='warn'
)

c_lambd = 1/np.sqrt(8)
for it in range(200):
    old_params = get_params()
    while True:
    # if True:
        for op, p in zip(old_params, params):
            p.set_value(op)

        t_r = train(c_lambd)
        e_v = eva()
        rho = (t_r[1]-e_v[0])/t_r[0]

        print(round(c_lambd, 5), round(rho, 2))
        RATE = 1.50
        if rho < 0.25:
            c_lambd *= RATE
        elif rho > 0.75:
            c_lambd /= RATE
        else:
            # pass
            break

    print(round(c_lambd, 9), round(rho, 2), t_r[1]-e_v[0], t_r)
