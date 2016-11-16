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
# import theano.sandbox.cuda.cusolver

from theano_utils import floatX, solve_sym_pos, rank, shared_empty
from loss import (get_pred, get_loss, get_loss_samples, get_regularizer,
                  get_error, get_total_loss)
# from cusolver import gpu_solve


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
        self.rand_outc_d = shared_empty(3)
        # ---

        self.rand_outc = T.tensor3('rand_outc')
        self.lambd_inv = T.scalar('lambd_inv')

        self.c_lambd_inv = 1e-3
        self.over_sampling = 1

        # -- target def --
        # self.loss_samples = get_loss_samples(self.model.a, self.outc)
        self.loss_samples = 0
        for i in range(self.over_sampling):
            self.loss_samples += get_loss_samples(self.model.a, self.rand_outc[i] + theano.gradient.consider_constant(self.model.a))
        self.loss = get_loss(self.model.a, self.outc)
        self.err = get_error(get_pred(self.model.a), self.y)

        self.grad = []
        self.updates = OrderedDict()


        for p in self.model.params:
            self.grad += [T.grad(self.loss, p)]
        self.grad_vec = T.concatenate([g.flatten() for g in self.grad])

        if True and 'Fisher':
            self.grad2d = []
            for p in self.model.params:
                self.grad2d += [T.jacobian(self.loss_samples, p)]
                if self.grad2d[-1].ndim == 2:
                    self.grad2d[-1] = self.grad2d[-1].dimshuffle(0, 1, 'x')

            self.grad2d_vec = T.concatenate([g.flatten(2).T for g in self.grad2d]).T

            # tensor wise: F_p,i,j = sum_k grad2d[p,i,k]*grad2d[p,k,j]
            # F = T.batched_dot(grad2d_vec.dimshuffle(0, 1, 'x'), grad2d_vec.dimshuffle(0, 'x', 1))
            # F = T.mean(F, 0)
            self.F = T.dot(self.grad2d_vec.T, self.grad2d_vec)/T.cast(self.grad2d_vec.shape[0], theano.config.floatX)/self.over_sampling
        elif 'GN':
            self.grad2d = []
            for p in self.model.params:
                self.grad2d += [T.jacobian(self.model.a.flatten(), p)]
                new_shape = (self.model.a.shape[0], self.model.a.shape[1], -1)
                self.grad2d[-1] = self.grad2d[-1].reshape(new_shape)


            self.grad2d_vec = T.concatenate([g.flatten(3) for g in self.grad2d], 2)
            # self.F = T.mean(T.batched_dot(self.grad2d_vec.dimshuffle(0, 2, 1),
            #                               self.grad2d_vec.dimshuffle(0, 1, 2)), axis=0)
            self.F = T.tensordot(self.grad2d_vec.dimshuffle(0, 2, 1),
                                 self.grad2d_vec.dimshuffle(0, 1, 2), [(0, 2), (0, 1)])/T.cast(self.grad2d_vec.shape[0], theano.config.floatX)

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

        self.print_pls = []
        self.print_pls += [self.F.shape, rank(self.F*10000)]
        self.print_pls += [T.mean(self.grad_vec**2)**0.5]

        self.train = theano.function(
            inputs=[self.lambd_inv],
            outputs=[self.est_loss, self.loss, self.err] + self.print_pls,
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
        self.rand_outc_d.set_value(floatX(nprng.randn(self.over_sampling, *outc.shape)))

        old_params = self.get_params()
        while True:
            for op, p in zip(old_params, self.model.params):
                p.set_value(op)

            """
            v = T.vector('v')
            get_Fv = theano.function(
                inputs=[v],
                outputs=[self.F.dot(v)],
                givens={
                    self.x: self.x_d,
                    self.outc: self.outc_d
                },
                allow_input_downcast=True
            )

            grad_at = theano.function(
                inputs=[],
                outputs=sum(([T.grad(self.loss, p)] for p in self.model.params), []),
                givens={
                    self.x: self.x_d,
                    self.outc: self.outc_d
                },
                allow_input_downcast=True
            )
            grads0 = grad_at()

            vec = []

            EPS = 1e-5
            for p in self.model.params:
                vec += [nprng.randn(*p.get_value().shape).astype(theano.config.floatX)]
                p.set_value(p.get_value()+vec[-1]*EPS)
            grads1 = grad_at()

            vec_vec = np.concatenate([p.flatten() for p in vec])
            F_vec = get_Fv(vec_vec)
            F_vec_vec = np.concatenate([f.flatten() for f in F_vec])

            grads0_vec = np.concatenate([p.flatten() for p in grads0])
            grads1_vec = np.concatenate([p.flatten() for p in grads1])

            F_vec_emp = (grads1_vec-grads0_vec)/EPS

            print(np.mean(F_vec_emp**2)**0.5, np.mean(F_vec_vec**2)**0.5)
            print(np.max(np.abs(F_vec_emp-F_vec_vec)))

            exit(0)
            """

            try:
                t_r = self.train(self.c_lambd_inv)
            except numpy.linalg.linalg.LinAlgError:
                t_r = [1e20, 1e10] + [None] * 4

            e_v = self.eva()
            delta_ll = t_r[1] - e_v[0]
            rho = delta_ll/float(t_r[0])

            print()
            print('lambda:', round(self.c_lambd_inv, 7), 'rho:', round(rho, 2), 'old loss:',  t_r[1], 'new loss:', e_v[0])
            RATE = 3.00
            if rho < 0:
                self.c_lambd_inv *= RATE * 2
                continue
            elif rho < 0.25:
                self.c_lambd_inv *= RATE
            elif rho > 0.75:
                self.c_lambd_inv /= RATE
            else:
                pass
            break

        return {'rho': rho, 'est_loss': t_r[0], 'loss': t_r[1], 'err': t_r[2], 'shape': t_r[3], 'rank': t_r[4], 'grad_mean': t_r[5], 'delta_ll': delta_ll}
