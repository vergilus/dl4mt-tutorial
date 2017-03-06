import numpy
import os

import numpy
import os

from nmt import train

def main(job_id, params):
    print params
    validerr = train(saveto=params['model'][0],
                     reload_=params['reload'][0],
                     dim_word=params['dim_word'][0],
                     dim=params['dim'][0],
                     n_words=params['n-words'][0],
                     n_words_src=params['n-words'][0],
                     
                     buffer_size=params['buffer_size'][0],
                     buffer_dim=params['buffer_size'][1],
                     
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     optimizer=params['optimizer'][0],
                     patience=1000,
                     maxlen=50,
                     batch_size=80,
                     valid_batch_size=80,
                     validFreq=100,
                     dispFreq=10,
                     saveFreq=100,
                     sampleFreq=100,
                     datasets=['../data/cn.txt.shuf',
                               '../data/en.txt.shuf'],
                     valid_datasets=['../data/MT02.cn.dev',
                                     '../data/MT02.en.dev'],
                     dictionaries=['../data/cn.txt.pkl',
                                   '../data/en.txt.pkl'],
                     use_dropout=params['use-dropout'][0],
                     overwrite=True
                     )
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['model_memDec.npz'],
        'dim_word': [512],
        'dim': [1024],
        'n-words': [30000],
        'buffer_size':[8,1024],
        'optimizer': ['adadelta'],
        'decay-c': [0.],
        'clip-c': [1.],
        'use-dropout': [True],
        'learning-rate': [0.0001],
        'reload': [True]})
