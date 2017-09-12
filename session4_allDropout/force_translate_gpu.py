'''
Translates a source file using a translation model.
'''
import argparse

import numpy
import theano
import json,io

import cPickle as pkl

from nmt import (build_force_sampler, load_params,
                 init_params, init_tparams)

from multiprocessing import Process, Queue
from numpy.distutils.core import _dict_append

def print_matrix(matrix):
    for ti, vector in enumerate(matrix):
        for si, w in enumerate(vector):
            print w,',',
        print
    return  

def main(model, 
         dictionary, dictionary_target, 
         source_file, reference_file,
         chr_level=False):

    print 'load model model_options'
    
    with open('%s.pkl' % model, 'rb') as f:
        options = pkl.load(f)

    print 'load source dictionary and invert'
    with open(dictionary, 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    print 'load target dictionary and invert'
    with open(dictionary_target, 'rb') as f:
        word_dict_trg = pkl.load(f)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk
    word_idict_trg[0] = '<eos>'
    word_idict_trg[1] = 'UNK'

    # utility function
    def _seqs2sen(seqs,_dict):
        sen = []
        for w in seqs:
            if w == 0:
                continue
            elif w < 0:
                continue
            sen.append(_dict[w])
        return ' '.join(sen) 

    def _send_jobs(fname, _dict, _n_words):# translate source sentence into source indices
        sourceIndices = []
        source = []
        with open(fname, 'r') as f:
            for idx, line in enumerate(f):
                if chr_level:
                    words = list(line.decode('utf-8').strip())
                else:
                    words = line.strip().split()
                x = map(lambda w: _dict[w] if w in _dict else 1, words)
                x = map(lambda ii: ii if ii < _n_words else 1, x)
                x += [0]
                sourceIndices.append(x)
                source.append(line)
        return sourceIndices , source

    print 'Force Translating ', source_file, '...'
    print 'Prepare data...',
    ret = _send_jobs(source_file,word_dict,options['n_words_src'])
    sourceIndices = ret[0]
    source = ret[1]
    
    ret_ref = _send_jobs(reference_file,word_dict_trg,options['n_words'])
    targetIndices = ret_ref[0]
    target = ret_ref[1]

    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    use_noise = theano.shared(numpy.float32(0.))

    # allocate model parameters
    params = init_params(options)

    # load model parameters and set theano shared variables
    params = load_params(model, params)
    tparams = init_tparams(params)

    # word index
    force_record = build_force_sampler(tparams, options, use_noise)

    def _translate(seq, trg_seq):
        # sample given an input sequence and obtain translated result
        sampleData = force_record(numpy.array(seq).reshape([len(seq), 1]),
                                  numpy.array(trg_seq).reshape([len(trg_seq), 1]))
        
        alpha_buffer_record=sampleData[0]
        attention_record=sampleData[1]
        
        if alpha_buffer_record is None:
            buffer_weight=None
        else:
            buffer_weight=alpha_buffer_record.reshape([len(trg_seq), options['buffer_size']])
            
        if attention_record is None:
            attention=None
        else:
            attention=attention_record.reshape([len(trg_seq), len(seq)])
            
        return buffer_weight, attention

    idx = 0
    print 'Done, translating...'
    for x, sSen, y in zip(sourceIndices, source, targetIndices):
        transData = _translate(x,y)
        
        buffer_weight=transData[0]
        attention=transData[1]
        
        print 'Sen ',idx, ':', sSen # source sentence
        tSen=_seqs2sen(y,word_idict_trg)
        print 'translation:', tSen # target sentence
        idx += 1
        print 'buffer_weight:'
        print_matrix(buffer_weight)
        print 'attention:'
        print_matrix(attention)
    
    print 'Done'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', action="store_true", default=False)
    parser.add_argument('model', type=str)
    parser.add_argument('dictionary', type=str)
    parser.add_argument('dictionary_target', type=str)
    parser.add_argument('source', type=str)
    parser.add_argument('reference', type=str)

    args = parser.parse_args()

    main(args.model, args.dictionary, args.dictionary_target, 
         args.source,args.reference,
         chr_level=args.c)
