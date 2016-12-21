import timeit
from collections import OrderedDict

import numpy as np
import numpy.linalg
import numpy.random as nprng

import theano
import theano.gradient
import theano.tensor.slinalg
import theano.tensor as T
import theano.printing
from theano.tensor.shared_randomstreams import RandomStreams
# import theano.sandbox.cuda.cusolver

from theano_utils import floatX, solve_sym_pos, rank, shared_empty
from loss import (get_pred, get_loss, get_loss_samples, get_regularizer,
                  get_error, get_total_loss)
# from cusolver import gpu_solve


class GDOptimizer:
    def __init__(self, model, algo="sgd"):
        self.model = model
        self.algo = algo

        self.x = self.model.x
        self.y = T.ivector('y')
        self.outc = T.matrix('outc')

        # due to theano bugs
        self.x_d = shared_empty(2)
        self.y_d = shared_empty(1, dtype="int32")
        self.outc_d = shared_empty(2)
        # ---

        # -- target def --
        self.loss = get_loss(self.model.a, self.outc)
        self.err = get_error(get_pred(self.model.a), self.y)

        self.grad_vec = T.concatenate([T.grad(self.loss, p).flatten() for p in self.model.params])

        srng = RandomStreams(seed=234)
        self.grad = {p: T.grad(self.loss, p) for p in self.model.params}

        for p in self.grad:
            self.grad[p] += srng.normal(p.shape)*1e-4

        self.updates = OrderedDict()

        if self.algo == 'gd':
            self.rate = 10.
            for p in self.model.params:
                self.updates[p] = p - self.rate * self.grad[p]
        elif self.algo == 'adagrad':
            self.rate = 5e-2
            eps = 1e-6
            for p in self.model.params:
                value = p.get_value(borrow=True)
                hist = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=p.broadcastable)
                hist_n = hist + self.grad[p]**2
                self.updates[hist] = hist_n
                self.updates[p] = p - self.rate * self.grad[p] / T.sqrt(hist + eps)
        elif self.algo == 'rmsprop':
            self.rate = 1e-2
            eps = 1e-6
            rho = 0.7
            for p in self.model.params:
                value = p.get_value(borrow=True)
                hist = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=p.broadcastable)
                hist_n = rho * hist + (1 - rho) * self.grad[p]**2
                self.updates[hist] = hist_n
                self.updates[p] = p - self.rate * self.grad[p] / T.sqrt(hist + eps)
        elif self.algo == 'nag':
            self.rate = 10
            mu = 0.2
            for p in self.model.params:
                value = p.get_value(borrow=True)
                vel = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=p.broadcastable)
                x = mu * vel + self.rate * self.grad[p]
                self.updates[vel] = x
                self.updates[p] = p - self.rate * self.grad[p] - mu * x
        elif self.algo == 'adam':
            self.rate = 4e-2
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            one = T.constant(1)
            t_prev = theano.shared(np.asarray(0, dtype=theano.config.floatX))
            t = t_prev + 1
            a_t = self.rate*T.sqrt(one-beta2**t)/(one-beta1**t)

            for p in self.model.params:
                value = p.get_value(borrow=True)

                m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=p.broadcastable)
                v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=p.broadcastable)

                m_t = beta1*m_prev + (one-beta1)*self.grad[p]
                v_t = beta2*v_prev + (one-beta2)*self.grad[p]**2
                step = a_t*m_t/(T.sqrt(v_t) + eps)

                self.updates[m_prev] = m_t
                self.updates[v_prev] = v_t
                self.updates[p] = p - step
            self.updates[t_prev] = t

        self.print_pls = []
        self.print_pls += [T.mean(self.grad_vec**2)**0.5]

        self.train = theano.function(
            inputs=[],
            outputs=[self.loss, self.err] + self.print_pls,
            updates=self.updates,
            givens={
                self.x: self.x_d,
                self.y: self.y_d,
                self.outc: self.outc_d,
            },
            on_unused_input='warn',
            allow_input_downcast=True
        )

        self.eva = theano.function(
            inputs=[],
            outputs=[self.loss],
            givens={
                self.x: self.x_d,
                self.outc: self.outc_d
            },
            on_unused_input='warn',
            allow_input_downcast=True
        )

    def step(self, X, y, outc):
        self.x_d.set_value(X)
        self.y_d.set_value(y)
        self.outc_d.set_value(outc)

        t_r = self.train()
        e_v = self.eva()
        delta_ll = t_r[0] - e_v[0]

        print()
        print('old loss:',  t_r[0], 'new loss:', e_v[0])

        return {'loss': t_r[0], 'err': t_r[1], 'grad_mean': t_r[2], 'delta_ll': delta_ll}
