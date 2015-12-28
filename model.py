import theano
import theano.tensor as T
import numpy
from utils import param_init, repeat_x
from theano.tensor.nnet import categorical_crossentropy


class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = param_init().uniform((n_in, n_out))
        # initialize the baises b as a vector of n_out 0s
        self.b = param_init().constant((n_out, ))

        # compute vector of class-membership probabilities in symbolic form
        energy = theano.dot(input, self.W) + self.b
        if energy.ndim == 3:
            energy_exp = T.exp(energy - T.max(energy, 2, keepdims=True))
            pmf = energy_exp / energy_exp.sum(2, keepdims=True)
        else:
            pmf = T.nnet.softmax(energy)

        self.p_y_given_x = pmf

        # compute prediction as class whose probability is maximal in
        # symbolic form

        # parameters of the model
        self.params = [self.W, self.b]

    def cost(self, targets, mask=None):
        prediction = self.p_y_given_x

        if prediction.ndim == 3:
        # prediction = prediction.dimshuffle(1,2,0).flatten(2).dimshuffle(1,0)
            prediction_flat = prediction.reshape(((prediction.shape[0] *
                                                prediction.shape[1]),
                                                prediction.shape[2]), ndim=2)
            targets_flat = targets.flatten()
            mask_flat = mask.flatten()
            ce = categorical_crossentropy(prediction_flat, targets_flat) * mask_flat
        else:
            assert mask is None
            ce = categorical_crossentropy(prediction, targets)
        return T.sum(ce)

    def errors(self, y):
        y_pred = T.argmax(self.p_y_given_x, axis=-1)
        if y.ndim == 2:
            y = y.flatten()
            y_pred = y_pred.flatten()
        return T.sum(T.neq(y, y_pred))



class GRU(object):

    def __init__(self, n_in, n_hids, with_contex=False, **kwargs):
        self.n_in = n_in
        self.n_hids = n_hids
        self.with_contex = with_contex
        if self.with_contex:
            self.c_hids = kwargs.pop('c_hids', n_hids)
        self._init_params()

    def _init_params(self):
        n_in = self.n_in
        n_hids = self.n_hids
        size_xh = (n_in, n_hids)
        size_hh = (n_hids, n_hids)
        self.W_xz = param_init().uniform(size_xh)
        self.W_xr = param_init().uniform(size_xh)
        self.W_xh = param_init().uniform(size_xh)

        self.W_hz = param_init().orth(size_hh)
        self.W_hr = param_init().orth(size_hh)
        self.W_hh = param_init().orth(size_hh)

        self.b_z = param_init().constant((n_hids,))
        self.b_r = param_init().constant((n_hids,))
        self.b_h = param_init().constant((n_hids,))

        self.params = [self.W_xz, self.W_xr, self.W_xh,
                       self.W_hz, self.W_hr, self.W_hh,
                       self.b_z, self.b_r, self.b_h]

        if self.with_contex:
            size_ch = (self.c_hids, self.n_hids)
            self.W_cz = param_init().uniform(size_ch)
            self.W_cr = param_init().uniform(size_ch)
            self.W_ch = param_init().uniform(size_ch)
            self.W_c_init = param_init().uniform(size_ch)

            self.params = self.params + [self.W_cz, self.W_cr,
                                         self.W_ch, self.W_c_init]

    def _step_forward_with_context(self, x_t, x_m, h_tm1, c_z, c_r, c_h):
        '''
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
        c_x: contex of the rnn
        '''
        z_t = T.nnet.sigmoid(T.dot(x_t, self.W_xz) +
                             T.dot(h_tm1, self.W_hz) + c_z + self.b_z)

        r_t = T.nnet.sigmoid(T.dot(x_t, self.W_xr) +
                             T.dot(h_tm1, self.W_hr) + c_r + self.b_r)

        can_h_t = T.tanh(T.dot(x_t, self.W_xh) +
                         r_t * T.dot(h_tm1, self.W_hh) + c_h +
                         self.b_h)
        h_t = (1 - z_t) * h_tm1 + z_t * can_h_t

        h_t = x_m[:, None] * h_t + (1. - x_m[:, None])*h_tm1
        return h_t


    def _step_forward(self, x_t, x_m, h_tm1):
        '''
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
        c_x: contex of the rnn
        '''
        z_t = T.nnet.sigmoid(T.dot(x_t, self.W_xz) +
                             T.dot(h_tm1, self.W_hz) + self.b_z)

        r_t = T.nnet.sigmoid(T.dot(x_t, self.W_xr) +
                             T.dot(h_tm1, self.W_hr) + self.b_r)

        can_h_t = T.tanh(T.dot(x_t, self.W_xh) +
                         r_t * T.dot(h_tm1, self.W_hh) +
                         self.b_h)
        h_t = (1 - z_t) * h_tm1 + z_t * can_h_t

        h_t = x_m[:, None] * h_t + (1. - x_m[:, None])*h_tm1
        return h_t

    def apply(self, state_below, mask_below, init_state=None, context=None):
        if state_below.ndim == 3:
            batch_size = state_below.shape[1]
            n_steps = state_below.shape[0]
        else:
            raise NotImplementedError


        if self.with_contex:
            if init_state is None:
                init_state = T.tanh(theano.dot(context, self.W_c_init))
            c_z = theano.dot(context, self.W_cz)
            c_r = theano.dot(context, self.W_cr)
            c_h = theano.dot(context, self.W_ch)
            non_sequences = [c_z, c_r, c_h]
            rval, updates = theano.scan(self._step_forward_with_context,
                                        sequences=[state_below, mask_below],
                                        outputs_info=[init_state],
                                        non_sequences=non_sequences,
                                        n_steps=n_steps
                                        )

        else:
            if init_state is None:
                init_state = T.alloc(numpy.float32(0.), batch_size, self.n_hids)
            rval, updates = theano.scan(self._step_forward,
                                        sequences=[state_below, mask_below],
                                        outputs_info=[init_state],
                                        n_steps=n_steps
                                        )
        self.output = rval
        return self.output

    def merge_out(self, state_below, mask_below, context=None):
        hiddens = self.apply(state_below, mask_below, context=context)
        if context is None:
            msize = self.n_in + self.n_hids
            osize = self.n_hids
            combine = T.concatenate([state_below, hiddens], axis=2)
        else:
            msize = self.n_in + self.n_hids + self.c_hids
            osize = self.n_hids
            n_times = state_below.shape[0]
            m_context = repeat_x(context, n_times)
            combine = T.concatenate([state_below, hiddens, m_context], axis=2)

        self.W_m = param_init().uniform((msize, osize*2))
        self.b_m = param_init().constant((osize*2,))
        self.params += [self.W_m, self.b_m]

        merge_out = theano.dot(combine, self.W_m) + self.b_m
        merge_max = merge_out.reshape((merge_out.shape[0],
                                       merge_out.shape[1],
                                       merge_out.shape[2]/2,
                                       2), ndim=4).max(axis=3)
        return merge_max * mask_below[:, :, None]


class lookup_table(object):
    def __init__(self, embsize, vocab_size):
        self.W = param_init().uniform((vocab_size, embsize))
        self.params = [self.W]
        self.vocab_size = vocab_size
        self.embsize = embsize

    def apply(self, indices):
        outshape = [indices.shape[i] for i
                    in range(indices.ndim)] + [self.embsize]

        return self.W[indices.flatten()].reshape(outshape)


class auto_encoder(object):
    def __init__(self, sentence, sentence_mask, vocab_size, n_in, n_hids, **kwargs):
        layers = []

        batch_size = sentence.shape[1]
        encoder = GRU(n_in, n_hids, with_contex=False)
        layers.append(encoder)

        if 'table' in kwargs:
            table = kwargs['table']
        else:
            table = lookup_table(n_in, vocab_size)
        # layers.append(table)

        state_below = table.apply(sentence)
        context = encoder.apply(state_below, sentence_mask)[-1]

        decoder = GRU(n_in, n_hids, with_contex=True)
        layers.append(decoder)

        decoder_state_below = table.apply(sentence[:-1])
        hiddens = decoder.merge_out(decoder_state_below,
                                    sentence_mask[:-1], context=context)

        logistic_layer = LogisticRegression(hiddens, n_hids, vocab_size)
        layers.append(logistic_layer)

        self.cost = logistic_layer.cost(sentence[1:],
                                        sentence_mask[1:])/batch_size
        self.output = context
        self.params = []
        for layer in layers:
            self.params.extend(layer.params)









