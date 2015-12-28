import logging
logging.basicConfig(level=logging.INFO)
import os
from collections import Counter
import configurations

import cPickle as pickle
logger = logging.getLogger("preprocess")

class PrepareData:
    def __init__(self, train_file, train_dic,
                 bos_token, eos_token,
                 vocabsize=1000, unk_id=0,
                 unk_token='<unk>', seq_len=50, **kwargs):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.filename = train_file
        self.vocabsize = vocabsize
        self.unk_id = unk_id
        self.unk_token = unk_token
        self.seq_len = seq_len

        dic = self._creat_dic()
        f = open(train_dic, 'wb')
        pickle.dump(dic, f)
        logger.info('dump train dict {} has been dumped'.format(train_dic))

    def _creat_dic(self):
        if os.path.isfile(self.filename):
            train = open(self.filename)
        else:
            logger.warning("file name {} not exist".format(self.filename))

        seq_len = self.seq_len
        worddict={}
        worddict[self.unk_token] = self.unk_id

        counter = Counter()
        for line in train:
            line_vec = line.strip().split()
            if isinstance(seq_len, int):
                line_vec = line_vec[:seq_len]
            counter.update(line_vec)

        index = self.unk_id + 1
        if self.unk_token in counter:
            del counter[self.unk_token]
        for word, c in counter.most_common(self.vocabsize-3):
            worddict[word] = index
            index += 1

        if self.bos_token:
            worddict[self.bos_token] = len(worddict)
        if self.eos_token:
            worddict[self.eos_token] = len(worddict)

        #assert len(worddict) == self.vocabsize

        return worddict

if __name__ =="__main__":
    configuration = getattr(configurations, 'get_config_penn')()
    prepare = PrepareData(**configuration)
