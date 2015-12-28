def get_config_penn():
    config = {}
    config['seq_len'] = 50
    config['nhids'] = 500
    config['nemb'] = 500
    config['batch_size'] = 100
    config['vocabsize'] = 10001

    #####################
    datadir = './data/'
    config['train_file'] = datadir + 'train'
    config['valid_file'] = datadir + 'valid'
    config['test_file'] = datadir + 'test'
    config['train_dic'] = datadir + 'train_dic.pkl'
    config['unk_id'] = 0 # The value is fixed, dont change it.
    config['bos_token'] = '<s>'
    #config['eos_token'] = '</s>'
    config['eos_token'] = None
    config['unk_token'] = '<unk>'


    return config

