import theano
import theano.tensor as T
from model import GRU, lookup_table, LogisticRegression
from utils import adadelta
from stream import DStream
import numpy
import logging
logger = logging.getLogger(__name__)

class language_model(object):

    def __init__(self, vocab_size, n_in, n_hids, **kwargs):
        self.n_in = n_in
        self.n_hids = n_hids
        self.vocab_size = vocab_size

        self.params = []
        self.layers = []

    def apply(self, sentence, sentence_mask):

        src = sentence[:-1]
        src_mask = sentence_mask[:-1]
        tgt = sentence[1:]
        tgt_mask = sentence_mask[1:]
        table = lookup_table(self.n_in, self.vocab_size)
        state_below = table.apply(src)
        self.layers.append(table)

        rnn = GRU(self.n_in, self.n_hids)
        hiddens = rnn.merge_out(state_below, src_mask)
        self.layers.append(rnn)

        logistic_layer = LogisticRegression(hiddens, self.n_hids, self.vocab_size)
        self.layers.append(logistic_layer)

        self.cost = logistic_layer.cost(tgt, tgt_mask)

        for layer in self.layers:
            self.params.extend(layer.params)

        self.L2 = sum(T.sum(item ** 2) for item in self.params)
        self.L1 = sum(T.sum(abs(item)) for item in self.params)

def test(test_fn , tst_stream):
    sums = 0
    case = 0
    for sentence, sentence_mask in tst_stream.get_epoch_iterator():
        cost = test_fn(sentence.T, sentence_mask.T)
        sums += cost[0]
        case += sentence_mask[1:].sum()
    ppl = numpy.exp(sums/case)
    logger.info('ppl : {}'.format(ppl))

if __name__=='__main__':
    import configurations
    cfig = getattr(configurations, 'get_config_penn')()
    sentence = T.lmatrix()
    sentence_mask = T.matrix()
    lm = language_model(cfig['vocabsize'], cfig['nemb'], cfig['nhids'])
    lm.apply(sentence, sentence_mask)

    cost_sum = lm.cost
    cost_mean = lm.cost/sentence.shape[1]

    params = lm.params
    regular = lm.L1 * 1e-6 + lm.L2 * 1e-6

    grads = T.grad(cost_mean + regular, params)
    updates = adadelta(params, grads)
    ds = DStream(datatype='train', config=cfig)
    ts = DStream(datatype='test', config=cfig)

    fn = theano.function([sentence, sentence_mask], [cost_mean], updates=updates)
    test_fn = theano.function([sentence, sentence_mask], [cost_sum])
    for epoch in range(20):
        logger.info('{} epoch has been tackled;'.format(epoch))
        for data, mask in ds.get_epoch_iterator():
            c = fn(data.T, mask.T)
        test(test_fn, ts)


