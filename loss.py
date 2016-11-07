import theano
import theano.tensor as T


def get_pred(act):
    return T.argmax(act, axis=1)

def get_loss_samples(act, target):
    res = T.mean((act - target)**2/2, 1)
    return res


def get_loss(act, target):
    return T.mean(get_loss_samples(act, target))


def get_regularizer(params, w=1e-5):
    res = 0
    for p in params:
        res += T.sum(p**2)*w
    return res


def get_error(pred, target):
    return T.mean(T.neq(pred, target), dtype=theano.config.floatX)


def get_total_loss(act, target, params, w):
    return get_loss(act, target) + get_regularizer(params, w)
