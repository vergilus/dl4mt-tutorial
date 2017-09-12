import numpy
import os

from nmt import train

def main(job_id, params):
    print params
    
    bleu_params = {'valid_path':'../data/validate/',
                   'temp_dir':'../temp/',
                   'translate_script': 'translate_gpu.py',
                   'bleu_script': 'multi-bleu.perl'
                   }
    validerr = train(saveto=params['model'][0],
                     reload_=params['reload'][0],
                     dim=params['dim'][0],
                     n_words=params['n-words'][0],
                     n_words_src=params['n-words'][0],
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     optimizer=params['optimizer'][0],
                     patience=1000,
                     patience_bleu=100,
                     maxlen=100,
                     batch_size=100,
                     valid_batch_size=100,
                     validFreq=100,
                     dispFreq=10,
                     saveFreq=100,
                     sampleFreq=100,
                     datasets=['../data/cn.txt.sort',
                               '../data/en.txt.sort'],
                     valid_datasets=['../data/MT02.cn.dev',
                                     '../data/MT02.en.dev'],
                     dictionaries=['../data/cn.txt.pkl',
                                   '../data/en.txt.pkl'],
                     use_dropout=params['use-dropout'][0],
                     overwrite=True,
                     **bleu_params
                     )
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['model_allAtt.npz'],
        'dim': [512],
        'n-words': [30000],
        'optimizer': ['adam'],
        'decay-c': [0.],
        'clip-c': [1.],
        'use-dropout': [True],
        'learning-rate': [0.0001],
        'reload': [True]})
