import numpy as np

import scipy.linalg

import theano
import theano.tensor as T
import theano.gof


def floatX(a):
    return np.asarray(a, dtype=theano.config.floatX)


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
        print(A, b)
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
