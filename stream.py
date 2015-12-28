import logging
logging.basicConfig(level=logging.INFO)

from fuel.datasets import TextFile
from fuel.streams import DataStream
from fuel.schemes import ConstantScheme
from fuel.transformers import Batch, Padding, SortMapping, Unpack, Mapping

import cPickle as pickle

logger = logging.getLogger(__name__)


def _length(sentence):
        return len(sentence[0])

def DStream(datatype, config):

    if datatype=='train':
        filename = config['train_file']
    elif datatype == 'valid':
        filename = config['valid_file']
    elif datatype == 'test':
        filename = config['test_file']
    else:
        logger.error('wrong datatype, train, valid, or test')


    data = TextFile(files=[filename],
                    dictionary=pickle.load(open(config['train_dic'],'rb')),
                    unk_token=config['unk_token'],
                    level='word',
                    bos_token=config['bos_token'],
                    eos_token=config['eos_token'])

    data_stream = DataStream.default_stream(data)
    data_stream.sources = ('sentence',)


    # organize data in batches and pad shorter sequences with zeros
    batch_size = config['batch_size']
    data_stream = Batch(data_stream, iteration_scheme=ConstantScheme(batch_size*16))
    data_stream = Mapping(data_stream, SortMapping(_length))
    data_stream = Unpack(data_stream)
    data_stream = Batch(data_stream, iteration_scheme=ConstantScheme(batch_size))
    data_stream = Padding(data_stream)
    return data_stream

if __name__ == "__main__":
    import configurations
    configuration = getattr(configurations, 'get_config_penn')()
    ds = DStream(datatype='test', config=configuration)
    i = 1
    for data, mask in ds.get_epoch_iterator():
        print data[0], mask[0]
