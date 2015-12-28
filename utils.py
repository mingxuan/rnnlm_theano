import theano
import theano.tensor as T
import numpy
import logging
from itertools import izip

logger = logging.getLogger(__name__)

class param_init(object):

    def __init__(self,**kwargs):

        self.shared = kwargs.pop('shared', True)

    def param(self, size, init_type='uniform', **kwargs):
        try:
            func = getattr(self, init_type)
        except AttributeError:
            logger.error('AttributeError, {}'.format(init_type))
        else:
            param = func(size, **kwargs)
        if self.shared:
            param = theano.shared(value=param, borrow=True)
        return param

    def uniform(self, size, **kwargs):
        low = kwargs.pop('low', -6./sum(size))
        high = kwargs.pop('high', 6./sum(size))
        rng = kwargs.pop('rng', numpy.random.RandomState(1234))
        param = numpy.asarray(
            rng.uniform(low=low, high=high, size=size),
            dtype=theano.config.floatX)
        if self.shared:
            param = theano.shared(value=param, borrow=True)
        return param

    def normal(self, size, **kwargs):
        loc = kwargs.pop('loc', 0.)
        scale = kwargs.pop('scale', 0.05)
        rng = kwargs.pop('rng', numpy.random.RandomState(1234))
        param = numpy.asarray(
            rng.normal(loc=loc, scale=scale, size=size),
            dtype=theano.config.floatX)
        if self.shared:
            param = theano.shared(value=param, borrow=True)
        return param

    def constant(self, size, **kwargs):
        value = kwargs.pop('scale', 0.)
        param = numpy.ones(size, dtype=theano.config.floatX)*value
        if self.shared:
            param = theano.shared(value=param, borrow=True)
        return param

    def orth(self, size, **kwargs):
        scale = kwargs.pop('scale', 1.)
        rng = kwargs.pop('rng', numpy.random.RandomState(1234))
        if len(size) != 2:
            raise ValueError
        if size[0] == size[1]:
            M = rng.randn(*size).astype(theano.config.floatX)
            Q, R = numpy.linalg.qr(M)
            Q = Q * numpy.sign(numpy.diag(R))
            param = Q*scale
            if self.shared:
                param = theano.shared(value=param, borrow=True)
            return param
        else:
            M1 = rng.randn(size[0], size[0]).astype(theano.config.floatX)
            M2 = rng.randn(size[1], size[1]).astype(theano.config.floatX)
            Q1, R1 = numpy.linalg.qr(M1)
            Q2, R2 = numpy.linalg.qr(M2)
            Q1 = Q1 * numpy.sign(numpy.diag(R1))
            Q2 = Q2 * numpy.sign(numpy.diag(R2))
            n_min = min(size[0], size[1])
            param = numpy.dot(Q1[:, :n_min], Q2[:n_min, :]) * scale
            if self.shared:
                param = theano.shared(value=param, borrow=True)
            return param


def  repeat_x(x, n_times):
    # This is black magic based on broadcasting,
    # that's why variable names don't make any sense.
    a = T.shape_padleft(x)
    padding = [1] * x.ndim
    b = T.alloc(numpy.float32(1), n_times, *padding)
    out = a * b
    return out

def adadelta(parameters,gradients,rho=0.95,eps=1e-6):
    # create variables to store intermediate updates
    gradients_sq = [theano.shared(numpy.zeros(p.get_value().shape, dtype='float32')) for p in parameters ]
    deltas_sq = [theano.shared(numpy.zeros(p.get_value().shape, dtype='float32')) for p in parameters ]

    # calculates the new "average" delta for the next iteration
    gradients_sq_new = [ rho*g_sq + (1-rho)*(g**2) for g_sq,g in izip(gradients_sq,gradients) ]

    # calculates the step in direction. The square root is an approximation to getting the RMS for the average value
    deltas = [ (T.sqrt(d_sq+eps)/T.sqrt(g_sq+eps))*grad for d_sq,g_sq,grad in izip(deltas_sq,gradients_sq_new,gradients) ]

    # calculates the new "average" deltas for the next step.
    deltas_sq_new = [ rho*d_sq + (1-rho)*(d**2) for d_sq,d in izip(deltas_sq,deltas) ]

    # Prepare it as a list f
    gradient_sq_updates = zip(gradients_sq,gradients_sq_new)
    deltas_sq_updates = zip(deltas_sq,deltas_sq_new)
    parameters_updates = [ (p,T.clip(p - d, -15,15)) for p,d in izip(parameters,deltas) ]
    return gradient_sq_updates + deltas_sq_updates + parameters_updates


