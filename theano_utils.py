import numpy as np

import scipy.linalg

import theano
import theano.tensor as T
import theano.gof
import theano.compile
import theano.gradient


def floatX(a):
    return np.asarray(a, dtype=theano.config.floatX)


def scalar_floatX(a):
    return T.cast(a, theano.config.floatX)


def shared_empty(dim=2, dtype=None):
    if dtype is None:
        dtype = theano.config.floatX

    shp = tuple([1] * dim)
    return theano.shared(np.zeros(shp, dtype=dtype))


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
            z[0] = np.asarray(np.linalg.matrix_rank(x), dtype=x.dtype)
        except Exception:
            print('Failed to compute rank', x)
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


class SolveSymPos(theano.gof.Op):
    """
    Solve a positive symmetrical system of linear equations.
    """

    __props__ = ()

    def __init__(self):
        pass

    def __repr__(self):
        return 'SolveSymPos{%s}' % str(self._props())

    def make_node(self, A, b):
        A = T.as_tensor_variable(A)
        b = T.as_tensor_variable(b)
        assert A.ndim == 2
        assert b.ndim in [1, 2]
        otype = T.tensor(
            broadcastable=b.broadcastable,
            dtype=(A * b).dtype)
        return theano.gof.Apply(self, [A, b], [otype])

    def perform(self, node, inputs, output_storage):
        A, b = inputs
        # print(A, b)
        rval = scipy.linalg.solve(A, b, sym_pos=True)
        output_storage[0][0] = rval

    # computes shape of x where x = inv(A) * b
    def infer_shape(self, node, shapes):
        Ashape, Bshape = shapes
        rows = Ashape[1]
        if len(Bshape) == 1:  # b is a Vector
            return [(rows,)]
        else:
            cols = Bshape[1]  # b is a Matrix
            return [(rows, cols)]

solve_sym_pos = SolveSymPos()  # solve sym_pos

class FastKron(theano.gof.Op):
    """
    kron(A,B)
    """

    __props__ = ()

    def __init__(self):
        pass

    def __repr__(self):
        return 'FastKron{%s}' % str(self._props())

    def make_node(self, A, B):
        A = T.as_tensor_variable(A)
        B = T.as_tensor_variable(B)
        assert A.ndim in [2, 3]
        assert B.ndim in [2, 3]
        assert A.ndim == B.ndim
        otype = T.tensor(
            broadcastable=B.broadcastable,
            dtype=(A * B).dtype)
        return theano.gof.Apply(self, [A, B], [otype])

    def perform(self, node, inputs, output_storage):
        A, B = inputs
        if len(A.shape) == 2:
            res = np.kron(A, B)
        else:
            res = np.einsum('ijk,ibc->ijbkc', B, A).reshape((B.shape[0], A.shape[1]*B.shape[1], -1)).mean(0)[None, :]

        rval = res
        print(A.shape, B.shape, res.shape)
        output_storage[0][0] = rval

    # computes shape of x where x = inv(A) * b
    def infer_shape(self, node, shapes):
        Ashape, Bshape = shapes
        shape = (Ashape[-2]*Bshape[-2], Ashape[-1]*Bshape[-1])
        if len(Ashape) == 3:
            shape = (1,)+shape
        return [shape]

fast_kron = FastKron()

def native_kron(a, b):
    return T.batched_tensordot(b, a, [[], []]).dimshuffle(0, 1, 3, 2, 4).reshape((a.shape[0], a.shape[1]*b.shape[1], -1)).mean(0, keepdims=True)


class MyConsiderConstant(theano.compile.ViewOp):
    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)


    def grad(self, args, g_outs):
        return [g_out.zeros_like(g_out) for g_out in g_outs]


my_consider_constant = MyConsiderConstant()
