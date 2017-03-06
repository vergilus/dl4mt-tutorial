'''
Translates a source file using a translation model.
'''
import argparse

import numpy
import theano
import json,io

import cPickle as pkl

from nmt import (build_sampler, gen_sample, load_params,
                 init_params, init_tparams)

from multiprocessing import Process, Queue

def store_matrix(matrix, source, target, file):
    # sentences end with </s> symbol
    source.append("</s>") 
    target.append("</s>")
    links = []
    for ti, target_word_alignment in enumerate(matrix):
        for si,w in enumerate(target_word_alignment):
            links.append((target[ti], source[si], str(w),))
    json.dump(links,file,ensure_ascii=False)
    return 

def print_matrix(matrix):
    for ti, vector in enumerate(matrix):
        for si, w in enumerate(vector):
            print w,',',
        print
    return  

def main(model, dictionary, dictionary_target, source_file, saveto, k=8,
         normalize=False, n_process=5, chr_level=False):

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
    def _index2sens(caps):
        capsw = []
        for cc in caps:
            ww = []
            for w in cc:
                if w == 0:
                    continue
                # if w == -10000:
                #     ww.append('| NOTEND')
                #     continue
                elif w < 0:
                    # ww.append('|' +  str(w))
                    continue
                ww.append(word_idict_trg[w])
            capsw.append(' '.join(ww))
        return capsw

    def _seqs2sen(seqs):
        sen = []
        for w in seqs:
            if w == 0:
                continue
                # if w == -10000:
                #     ww.append('| NOTEND')
                #     continue
            elif w < 0:
                # ww.append('|' +  str(w))
                continue
            sen.append(word_idict_trg[w])
        return ' '.join(sen) 

    def _send_jobs(fname):# translate source sentence into indices
        sourceIndices = []
        source = []
        with open(fname, 'r') as f:
            for idx, line in enumerate(f):
                if chr_level:
                    words = list(line.decode('utf-8').strip())
                else:
                    words = line.strip().split()
                x = map(lambda w: word_dict[w] if w in word_dict else 1, words)
                x = map(lambda ii: ii if ii < options['n_words_src'] else 1, x)
                x += [0]
                sourceIndices.append(x)
                source.append(line)
        return sourceIndices , source

    print 'Translating ', source_file, '...'
    print 'Prepare data...',
    ret = _send_jobs(source_file)
    sourceIndices = ret[0]
    source = ret[1]

    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # allocate model parameters
    params = init_params(options)

    # load model parameters and set theano shared variables
    params = load_params(model, params)
    tparams = init_tparams(params)

    # word index
    f_init, f_next = build_sampler(tparams, options, trng, use_noise)

    def _translate(seq):
        # sample given an input sequence and obtain translated result
        sampleData = gen_sample(tparams, f_init, f_next,
                                   numpy.array(seq).reshape([len(seq), 1]),
                                   options, trng=trng, k=k, 
                                   maxlen=200,
                                   return_attention=True,
                                   analyseBuffer = True,
                                   stochastic = False, 
                                   argmax = False,
                                   normalize = normalize)
        sample=sampleData[0]
        score=sampleData[1]
        alpha_buffer_record=sampleData[2]
        attention_record=sampleData[3]
        
        # normalize scores according to sequence lengths
        if normalize:
            lengths = numpy.array([len(s) for s in sample])
            score = score / lengths
        sidx = numpy.argmin(score)
        if alpha_buffer_record is None:
            buffer_weight=None
        else:
            buffer_weight=alpha_buffer_record[sidx]
            
        if attention_record is None:
            attention=None
        else:
            attention=attention_record[sidx]
        return sample[sidx], buffer_weight, attention

    trans = []
    idx = 0
    print 'Done, translating...'
    for x , sSen in zip(sourceIndices , source):
        transData = _translate(x)
        y=transData[0]
        buffer_weight=transData[1]
        attention=transData[2]
        
        print 'Sen ',idx, ':',sSen # source sentence
        y = _seqs2sen(y)
        trans.append(y)
        print 'translation:', y  # translation result
        idx += 1
        print 'buffer_weight:'
        print_matrix(buffer_weight)
        print 'attention:'
        print_matrix(attention)
    


    with open(saveto, 'w') as f:
        print >>f, '\n'.join(trans)
    print 'Done'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=5)
    parser.add_argument('-p', type=int, default=5)
    parser.add_argument('-n', action="store_true", default=False)
    parser.add_argument('-c', action="store_true", default=False)
    parser.add_argument('model', type=str)
    parser.add_argument('dictionary', type=str)
    parser.add_argument('dictionary_target', type=str)
    parser.add_argument('source', type=str)
    parser.add_argument('saveto', type=str)

    args = parser.parse_args()

    main(args.model, args.dictionary, args.dictionary_target, args.source,
         args.saveto, normalize=args.n, n_process=args.p,
         chr_level=args.c)
