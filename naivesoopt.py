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

from theano_utils import (floatX, scalar_floatX, shared_empty, solve_sym_pos,
                          rank, fast_kron, native_kron, my_consider_constant)
from loss import (get_pred, get_loss, get_loss_samples, get_regularizer,
                  get_error, get_total_loss)
# from cusolver import gpu_solve


class NaiveSecondOrderOptimizer:
    def __init__(self, model, algo='fisher', c_lambd_inv=1e-3, rate=1.05,
                 over_sampling=1, rescale='momentum'):
        """ Init self.

        Args:
            model,
            algo,
            c_lambd_inv: Start value of \lambda regularizer (used in matrix
                inversion and in F*v computation).
            rate: Change per iteration for \lambda.
            over_sampling: For Fisher-like methods, use multiple random
                vectors per one sample from dataset.
            rescale: Can be either False, True or 'momentum'.

        Implemented algos:
            'gn' - Gauss-Newton matrix,
            'fisher' - Fisher matrix,
            'kr' - Khatri-Rao matrix,
            'kr_diag' - block-diagonal KR matrix.
        """
        self.model = model
        self.algo = algo

        self.x = self.model.x
        self.y = T.ivector('y')
        self.outc = T.matrix('outc')

        # due to theano bugs
        self.x_d = shared_empty(2)
        self.y_d = shared_empty(1, dtype='int32')
        self.outc_d = shared_empty(2)
        self.rand_outc_d = shared_empty(3)
        # ---

        self.rand_outc = T.tensor3('rand_outc')
        self.lambd_inv = T.scalar('lambd_inv')

        self.c_lambd_inv = c_lambd_inv
        self.rate = rate
        self.over_sampling = over_sampling
        self.rescale = rescale

        # -- target def --
        self.f_loss = 0
        self.f_loss_samples = 0
        for i in range(self.over_sampling):
            self.f_loss += get_loss(self.model.a, self.rand_outc[i] + my_consider_constant(self.model.a)) * scalar_floatX(self.model.a.shape[0])
            self.f_loss_samples += get_loss_samples(self.model.a, self.rand_outc[i] + my_consider_constant(self.model.a))

        self.loss = get_loss(self.model.a, self.outc)
        self.err = get_error(get_pred(self.model.a), self.y)

        self.updates = OrderedDict()

        self.grad = sum(([T.grad(self.loss, p)] for p in self.model.params), [])
        self.grad_vec = T.concatenate([g.flatten() for g in self.grad])

        def get_fisher_mat():
            grad2d = []
            for p in self.model.params:
                grad2d += [T.jacobian(self.f_loss_samples, p)]
                if grad2d[-1].ndim == 2:
                    grad2d[-1] = grad2d[-1].dimshuffle(0, 1, 'x')

            grad2d_vec = T.concatenate([g.flatten(2).T for g in grad2d]).T

            # tensor wise: F_p,i,j = sum_k grad2d[p,i,k]*grad2d[p,k,j]
            # just a slow reference implementation of what is below
            # F = T.mean(T.batched_dot(grad2d_vec.dimshuffle(0, 1, 'x'), grad2d_vec.dimshuffle(0, 'x', 1)), 0)/self.over_sampling
            F = T.dot(grad2d_vec.T, grad2d_vec)/T.cast(grad2d_vec.shape[0], theano.config.floatX)/self.over_sampling
            return F

        if self.algo == 'fisher':
            self.grad2d = []
            for p in self.model.params:
                self.grad2d += [T.jacobian(self.f_loss_samples, p)]
                if self.grad2d[-1].ndim == 2:
                    self.grad2d[-1] = self.grad2d[-1].dimshuffle(0, 1, 'x')

            self.grad2d_vec = T.concatenate([g.flatten(2).T for g in self.grad2d]).T

            # tensor wise: F_p,i,j = sum_k grad2d[p,i,k]*grad2d[p,k,j]
            # just a slow reference implementation of what is below
            # F = T.mean(T.batched_dot(grad2d_vec.dimshuffle(0, 1, 'x'), grad2d_vec.dimshuffle(0, 'x', 1)), 0)/self.over_sampling
            self.F = T.dot(self.grad2d_vec.T, self.grad2d_vec)/T.cast(self.grad2d_vec.shape[0], theano.config.floatX)/self.over_sampling
        elif self.algo == 'gn':
            self.grad2d = []
            for p in self.model.params:
                self.grad2d += [T.jacobian(self.model.a.flatten(), p)]
                new_shape = (self.model.a.shape[0], self.model.a.shape[1], -1)
                self.grad2d[-1] = self.grad2d[-1].reshape(new_shape)


            self.grad2d_vec = T.concatenate([g.flatten(3) for g in self.grad2d], 2)

            # just a slow reference implementation of what is below
            # self.F = T.mean(T.batched_dot(self.grad2d_vec.dimshuffle(0, 2, 1),
            #                               self.grad2d_vec.dimshuffle(0, 1, 2)), axis=0)

            self.F = T.tensordot(self.grad2d_vec.dimshuffle(0, 2, 1),
                                 self.grad2d_vec.dimshuffle(0, 1, 2), [(0, 2), (0, 1)])/T.cast(self.grad2d_vec.shape[0], theano.config.floatX)
        elif self.algo.startswith('kr'):
            self.grads = []
            # self.acts = [T.concatenate([self.model.x, T.ones((self.model.x.shape[0], 1))], axis=1)]
            self.acts = [self.model.x]
            for l in self.model.layers:
                cg = T.grad(self.f_loss, l.s)
                self.grads.append(cg)
                # self.acts.append(T.concatenate([l.a, T.ones((l.a.shape[0], 1))], axis=1))
                self.acts.append(l.a)

            self.G = []
            self.A = []
            self.F_block = []
            self.F = []

            cnt = T.cast(self.grads[0].shape[0], theano.config.floatX)
            for i in range(len(self.grads)):
                self.G += [[]]
                self.A += [[]]
                for j in range(len(self.grads)):
                    # self.G[-1] += [T.mean(T.batched_dot(self.grads[i].dimshuffle(0, 1, 'x'), self.grads[j].dimshuffle(0, 'x', 1)), 0).dimshuffle('x', 0, 1)]
                    # self.A[-1] += [T.mean(T.batched_dot(self.acts[i].dimshuffle(0, 1, 'x'), self.acts[j].dimshuffle(0, 'x', 1)), 0).dimshuffle('x', 0, 1)]

                    # self.G[-1] += [T.batched_dot(self.grads[i].dimshuffle(0, 1, 'x'), self.grads[j].dimshuffle(0, 'x', 1))]
                    # self.A[-1] += [T.batched_dot(self.acts[i].dimshuffle(0, 1, 'x'), self.acts[j].dimshuffle(0, 'x', 1))]

                    self.G[-1] += [self.grads[i].T.dot(self.grads[j]).dimshuffle('x', 0, 1)/cnt]
                    self.A[-1] += [self.acts[i].T.dot(self.acts[j]).dimshuffle('x', 0, 1)/cnt]

                    if self.algo.endswith('diag'):
                        self.G[-1][-1] *= float(i==j)
                        self.A[-1][-1] *= float(i==j)


            for i in range(len(self.grads)):
                self.F_block += [[]]
                for j in range(len(self.grads)):
                    # depends on whether you want to compute the real fisher with this or the kr approximation
                    # since numpy-base fast_kron somehow computes 3d tensors faster than theano

                    # cblock = fast_kron(self.A[i][j], self.G[i][j])
                    cblock = native_kron(self.A[i][j], self.G[i][j])

                    cblock = cblock.reshape(cblock.shape[1:], ndim=2)
                    self.F_block[i] += [cblock]
                self.F.append(T.concatenate(self.F_block[-1], axis=1))
            self.F = T.concatenate(self.F, axis=0)
            self.F = (self.F+self.F.T)/2


        self.Fdamp = self.F+T.identity_like(self.F)*self.lambd_inv

        # There're 3+ different ways of computing F^-1*v in theano,
        # and it seems like solve_sym_pos is quite neutral in terms
        # of performance + it throws an exception if the provided matrix
        # is singular.

        # self.new_grad_vec = theano.tensor.slinalg.solve(self.Fdamp, self.grad_vec.dimshuffle(0, 'x'))
        self.new_grad_vec = solve_sym_pos(self.Fdamp, self.grad_vec)
        # self.new_grad_vec = gpu_solve(self.Fdamp, self.grad_vec.dimshuffle(0, 'x'))

        pcount = sum(p.get_value().size for p in self.model.params)
        self.ch_history = theano.shared(np.zeros((pcount,), dtype=theano.config.floatX))

        if self.rescale == 'momentum':
            self.real_fish = get_fisher_mat() + T.identity_like(self.F)*self.lambd_inv

            FT = self.real_fish.dot(self.new_grad_vec)
            FM = self.real_fish.dot(self.ch_history)

            TFT = self.new_grad_vec.T.dot(FT)
            MFT = self.ch_history.T.dot(FT)
            MFM = self.ch_history.T.dot(FM)

            GT = self.grad_vec.T.dot(self.new_grad_vec)
            GM = self.grad_vec.T.dot(self.ch_history)


            tmp1 = T.stack([TFT.reshape(()), MFT.reshape(())], 0).dimshuffle('x', 0)
            tmp2 = T.stack([MFT.reshape(()), MFM.reshape(())], 0).dimshuffle('x', 0)

            A = T.concatenate([tmp1, tmp2], 0)
            A_pinv = T.nlinalg.MatrixPinv()(A)
            b = T.stack([GT.reshape(()), GM.reshape(())], 0).dimshuffle(0, 'x')

            res = A_pinv.dot(b).flatten()

            alpha = res[0]
            beta = res[1]

            self.new_grad_vec = self.new_grad_vec * alpha.reshape(()) + self.ch_history * beta.reshape(())
            self.F = self.real_fish

            self.updates[self.ch_history] = self.new_grad_vec
        elif self.rescale:
            self.real_fish = get_fisher_mat() + T.identity_like(self.F)*self.lambd_inv
            lin_fac = self.grad_vec.T.dot(self.new_grad_vec)
            quad_fac = self.new_grad_vec.T.dot(self.real_fish.dot(self.new_grad_vec))

            alpha = lin_fac/quad_fac
            beta = 0 * alpha

            self.new_grad_vec *= alpha.reshape(())
            self.F = self.real_fish
            # self.Fdamp = self.F+T.identity_like(self.F)*self.lambd_inv

        # alpha = T.as_tensor_variable(1)

        def _apply_gradient_vec(params, new_grad_vec, updates):
            new_grad = []
            offset = 0
            for p in params:
                pval = p.get_value()
                new_grad += [new_grad_vec[offset:offset+pval.size].reshape(pval.shape)]
                offset += pval.size

                updates[p] = p - new_grad[-1]

            return new_grad

        self.new_grad = _apply_gradient_vec(self.model.params, self.new_grad_vec, self.updates)

        self.get_params = theano.function(
            inputs=[],
            outputs=self.model.params,
            on_unused_input='warn'
        )

        self.quad_est_loss = self.new_grad_vec.T.dot(self.F.dot(self.new_grad_vec))/2
        self.est_loss = self.quad_est_loss + self.grad_vec.dot(self.new_grad_vec)

        self.print_pls = {}
        self.print_pls.update({'shape': self.F.shape[0], 'rank': rank(self.F*10000)})
        self.print_pls.update({'grad_mean': T.mean(self.grad_vec**2)**0.5})
        self.print_pls.update({'alpha': alpha, 'beta': beta})
        # self.print_pls += [self.F]
        # self.print_pls += [self.real_fish]

        self.train = theano.function(
            inputs=[self.lambd_inv],
            outputs=[self.est_loss, self.loss, self.err] + list(self.print_pls.values()),
            updates=self.updates,
            givens={
                self.x: self.x_d,
                self.y: self.y_d,
                self.outc: self.outc_d,
                self.rand_outc: self.rand_outc_d
            },
            on_unused_input='warn',
            allow_input_downcast=True,
            # profile=True
        )

        self.eva = theano.function(
            inputs=[],
            outputs=[self.loss, self.err],
            givens={
                self.x: self.x_d,
                self.y: self.y_d,
                self.outc: self.outc_d
            },
            on_unused_input='warn',
            allow_input_downcast=True
        )

    def step(self, X, y, outc):
        """Perform single train iteration.

        Args:
            X: input vectors
            y: target labels.
            outc: target vectors.

        Returns:
            Dict consisting of 'loss', 'err', 'est_loss', 'rho', 'delta_ll' and
            parameters from self.print_pls.

        """
        self.x_d.set_value(X)
        self.y_d.set_value(y)
        self.outc_d.set_value(outc)
        self.rand_outc_d.set_value(floatX(nprng.randn(self.over_sampling, *outc.shape)))

        old_params = self.get_params()
        while True:
            # reset params to saved
            for op, p in zip(old_params, self.model.params):
                p.set_value(op)

            try:
                t_r = self.train(self.c_lambd_inv)

                print_pls_vals = t_r[-len(self.print_pls):]
                self.print_pls_res = {k: v for k, v in zip(self.print_pls.keys(), print_pls_vals)}
            except numpy.linalg.linalg.LinAlgError:
                t_r = [1e20, 1e10, 10] + [None] * len(self.print_pls)
                self.print_pls_res = {k: None for k in self.print_pls.keys()}

            e_v = self.eva()
            delta_ll = t_r[1] - e_v[0]
            rho = delta_ll/float(t_r[0])

            print()
            print('lambda:', round(self.c_lambd_inv, 7), 'rho:', round(rho, 2), 'old loss:',  t_r[1], 'new loss:', e_v[0])
            if rho < 0:
                self.c_lambd_inv *= self.rate * 2
                continue
            elif rho < 0.5:
                self.c_lambd_inv *= self.rate
                # self.c_lambd_inv = min(self.c_lambd_inv, 0.02)
            elif rho > 0.5:
                self.c_lambd_inv /= self.rate
            else:
                pass
            break

        # self.train.profiler.print_summary()
        res = {'rho': rho, 'est_loss': t_r[0], 'loss': t_r[1], 'err': t_r[2], 'delta_ll': delta_ll}
        res.update(self.print_pls_res)

        return res

    def evaluate(X_test, y_test, outc_test):
        """Return loss and error for provided dataset.

        Args:
            X_test: input vectors,
            y_test: target labels,
            outc_test: target vectors.

        Returns:
            Dict consisting of 'test_loss', 'test_err'.
        """
        self.x_d.set_value(X_test)
        self.y_d.set_value(y_test)
        self.outc_d.set_value(outc_test)

        te_v = self.eva()
        test_loss = te_v[0]
        test_err = te_v[1]

        return {'test_loss': test_loss, 'test_err': test_err}

    def _check_gv_matrix_correctness(self):
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
