import timeit
from collections import OrderedDict

import numpy as np
import numpy.linalg
import numpy.random as nprng

import theano
import theano.tensor.slinalg
import theano.tensor as T
import theano.printing
import theano.sandbox.cuda.cusolver

from theano_utils import floatX, solve_sym_pos, rank, shared_empty
from loss import (get_pred, get_loss, get_loss_samples, get_regularizer,
                  get_error, get_total_loss)
from cusolver import gpu_solve


class NatGradOpt:
    def __init__(self, model):
        self.model = model

        self.x = self.model.x
        self.y = T.ivector('y')
        self.outc = T.matrix('outc')

        # due to theano bugs
        self.x_d = shared_empty(2)
        self.y_d = shared_empty(1, dtype="int32")
        self.outc_d = shared_empty(2)
        self.rand_outc_d = shared_empty(2)
        # ---

        self.rand_outc = T.matrix('rand_outc')
        self.lambd = T.scalar('lambd')
        self.lambd_inv = T.scalar('lambd_inv')

        self.c_lambd = 2.0 # 1/np.sqrt(8)
        self.c_lambd_inv = 1e-4

        # -- target def --
        # loss_samples = get_loss_samples(self.model.a, self.outc)
        self.loss_samples = get_loss_samples(self.model.a, self.rand_outc*self.lambd)
        self.loss = get_loss(self.model.a, self.outc)
        self.err = get_error(get_pred(self.model.a), self.y)

        self.grad = []
        self.grad2d = []
        self.updates = OrderedDict()

        self.print_pls = []

        for p in self.model.params:
            self.grad += [T.grad(self.loss, p)]
            self.grad2d += [T.jacobian(self.loss_samples, p)]
            if self.grad2d[-1].ndim == 2:
                self.grad2d[-1] = self.grad2d[-1].dimshuffle(0, 1, 'x')

        self.grad_vec = T.concatenate([g.flatten() for g in self.grad])
        self.grad2d_vec = T.concatenate([g.flatten(2).T for g in self.grad2d]).T

        # tensor wise: F_p,i,j = sum_k grad2d[p,i,k]*grad2d[p,k,j]
        # F = T.batched_dot(grad2d_vec.dimshuffle(0, 1, 'x'), grad2d_vec.dimshuffle(0, 'x', 1))
        # F = T.mean(F, 0)
        self.F = T.dot(self.grad2d_vec.T, self.grad2d_vec)/T.cast(self.grad2d_vec.shape[0], theano.config.floatX)
        self.print_pls += [self.F.dot(self.grad_vec).shape, self.F.shape, rank(self.F*10000), (self.F**2).trace()]

        self.Fdamp = self.F+T.identity_like(self.F)*self.lambd_inv
        # self.new_grad_vec = theano.tensor.slinalg.solve(self.Fdamp, self.grad_vec.dimshuffle(0, 'x'))
        self.new_grad_vec = solve_sym_pos(self.Fdamp, self.grad_vec)
        # self.new_grad_vec = gpu_solve(self.Fdamp*10000, self.grad_vec.dimshuffle(0, 'x')*10000)

        self.new_grad = []

        offset = 0
        for p in self.model.params:
            pval = p.get_value()
            self.new_grad += [self.new_grad_vec[offset:offset+pval.size].reshape(pval.shape)]
            offset += pval.size

            self.updates[p] = p - self.new_grad[-1]

        self.get_params = theano.function(
            inputs=[],
            outputs=self.model.params,
            on_unused_input='warn'
        )

        self.quad_est_loss = self.new_grad_vec.T.dot(self.F.dot(self.new_grad_vec))/2
        self.est_loss = self.quad_est_loss + self.grad_vec.dot(self.new_grad_vec)

        self.print_pls += [self.quad_est_loss, T.mean(self.grad_vec**2)**0.5]

        self.train = theano.function(
            inputs=[self.lambd, self.lambd_inv],
            outputs=[self.est_loss, self.loss, self.err, T.sum((self.model.a-self.outc)**2)/2] + self.print_pls,
            updates=self.updates,
            givens={
                self.x: self.x_d,
                self.y: self.y_d,
                self.outc: self.outc_d,
                self.rand_outc: self.rand_outc_d
            },
            on_unused_input='warn',
            allow_input_downcast=True
        )

        self.eva = theano.function(
            inputs=[],
            outputs=[self.loss, T.sum((self.model.a-self.outc)**2)/2],
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
        self.rand_outc_d.set_value(floatX(nprng.randn(*outc.shape)*np.sqrt(2)))

        old_params = self.get_params()
        what = nprng.random() < 0.5
        print('l' if what else 'i')
        while True:
        # if True:
            for op, p in zip(old_params, self.model.params):
                p.set_value(op)

            t_r = self.train(self.c_lambd, self.c_lambd_inv)
            e_v = self.eva()
            delta_ll = t_r[1] - e_v[0]
            rho = delta_ll/float(t_r[0])

            print(round(self.c_lambd, 5), round(self.c_lambd_inv, 5), e_v, round(rho, 2), t_r[1], e_v[0])
            RATE = 1.10
            if rho < 0.25:
                if what or True:
                    self.c_lambd *= RATE
                    # self.c_lambd_inv *= RATE
                else:
                    self.c_lambd_inv *= RATE**4
            elif rho > 0.75:
                if what and False:
                    self.c_lambd /= RATE
                else:
                    self.c_lambd /= RATE
                    self.c_lambd_inv /= RATE**4
            else:
                # pass
                break

        return rho, t_r, delta_ll
