'''
Build a neural machine translation model with soft attention
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import ipdb
import numpy
import copy

import os
import warnings
import sys
import time

from collections import OrderedDict
from data_iterator import TextIterator
from bleu_validator import BleuValidator

sys.setrecursionlimit(10000)
profile = False
floatX = theano.config.floatX
numpy_floatX = numpy.typeDict[floatX]

def zipp(params, tparams):
    # push parameters to Theano shared variables
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

def unzip(zipped):
    # pull parameters from Theano shared variables
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

def itemlist(tparams):
    # get the list of parameters: Note that tparams must be OrderedDict
    return [vv for kk, vv in tparams.iteritems()]

def dropout_layer(state_before, use_noise, trng, drop_rate=0.5):
    # dropout
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=1-drop_rate, n=1,
                                     dtype=state_before.dtype),
        state_before * (1-drop_rate))
    return proj

def normalize_layer(x,gamma,beta):
    epsilon=numpy_floatX(1e-8)
    if x.ndim==3:
        result=(x-x.mean(2)[:,:,None]) / tensor.sqrt((x.var(2)[:,:,None]+epsilon))
        result = gamma[None,None,:] * result +beta[None,None,:]
    else:
        result=(x-x.mean(1)[:,None]) / tensor.sqrt((x.var(1)[:,None]+epsilon))
        result = gamma[None,:] * result + beta[None,:]
    return result

def _p(pp, name):# make prefix-appended name
    return '%s_%s' % (pp, name)

def init_tparams(params):
    # initialize Theano shared variables according to the initial parameters
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

def load_params(path, params):
    # load parameters
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]

    return params

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'ffNorm': ('param_init_ffNorm', 'ffNorm_layer'),
          'fullAtt': ('param_init_multiAtt', 'multiAtt_layer'),
          }

def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))

# some utilities
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')

def tanh(x):
    return tensor.tanh(x)
def relu(x):
    return tensor.nnet.relu(x)
def linear(x):
    return x
def softmax(state_below, axis=-1):
    #numeric-stable softmax: for unlimited input through softmax for CE cost
    state_below -= state_below.max(axis=axis,keepdims=True)
    state_below -= tensor.log(tensor.exp(state_below).sum(axis=axis,keepdims=True))
    return tensor.exp(state_below)

def split(tensor, n_splits, split_size, axis=0):
    """
    Alternative implementation of 'theano.tensor.split'
    same as function concatenate
    :usage: 
        >>> a = theano.tensor.zeros(5,7,32)
        b = split(a, [32/8]*8, 8, axis=2)
    :parameter:
        - tensor: hyper tensor in shape=[5,7,32]
        - n_splits: size of every element n_splits=[8,8,8,8] split_size=4 axis=-1(equal to 'ndim-1')
        - split_size: the list size
        - axis: tensor will be split along this axis
    :retrun:
        - tensor_list: a list of tensor variable
    """
    if axis<0:
        axis += tensor.ndim
    tensor_list = []
    offset = 0
    # determine the result by indexing, save to tensor_list 
    for size in n_splits:
        indices=()
        for k in range(axis):
            indices += (slice(None), )
        indices += (slice(offset, offset+size), )
        for k in range(axis+1,tensor.ndim):
            indices += (slice(None), )
        offset += size
        tensor_list+= (tensor[indices],)
    
    return tensor_list

def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    if axis<0:
        axis += tensor_list[0].ndim
        
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out
# batch preparation
def prepare_data(seqs_x, seqs_y, maxlen=None, n_words_src=30000,
                 n_words=30000):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx]+1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.
    return x, x_mask, y, y_mask

def position_emb(seqs,dim):
    # the seqs(tensor.variable) shape: timesteps * nsamples
    # the return shape: timesteps * nsamples * dim
    # using scan over 
    timestep=seqs.shape[0]
    nsamples=seqs.shape[1]
    def slice(pos, i):
        # shape of pos is 1D
        # shape of i is a tensor vector:1*dim
        chunk_i=(i/2*2).astype('float32')
        temp=((pos/10000)*tensor.ones_like(i))**(chunk_i/dim)
        sym= -1*tensor.ones_like(i)
        result=0.5*(tensor.ones_like(i)+sym**i)*tensor.sin(temp)\
            +0.5*(tensor.ones_like(i)+sym**(i+1))*tensor.cos(temp)
        return result
    result,updates=theano.scan(slice,
                               sequences=[tensor.arange(timestep).astype('float32')],
                               non_sequences=[tensor.arange(dim).astype('int64')],
                               )
    # result shape: timestep*dim
    result=result.dimshuffle([0,'x',1])
    result=tensor.tile(result, [1,nsamples,1])
    return result

# feedforward layer
def param_init_fflayer(options, params, prefix='ff', 
                       nin=None, nout=None,ortho=True):
    if nin is None:
        nin = options['dim']
    if nout is None:
        nout = options['dim']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')
    return params

def fflayer(tparams, options, state_below, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    result=tensor.dot(state_below, tparams[_p(prefix,'W')])+tparams[_p(prefix,'b')]
    result=eval(activ)(result)
    return result
# feedforward layer with layer normalization
def param_init_ffNorm(options, params, prefix='ff_norm',
                      nin=None,nout=None,ortho=True):
    if nin is None:
        nin=options['dim']
    if nout is None:
        nout=options['dim']
    params[_p(prefix, 'W')]=norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b')]=numpy.zeros((nout,)).astype('float32')

    # for feedforward batch normalization
    params[_p(prefix, 'gamma')]=numpy.ones((nout,)).astype('float32')
    params[_p(prefix, 'beta')]=numpy.zeros((nout,)).astype('float32')
    return params

def ffNorm_layer(tparams, options, state_below, prefix='ff_norm', 
           activ='lambda x: tensor.nnet.relu(x)', epsilon=1e-8,
           **kwargs):
    # feed forward layer with residual connection and normalization
    result=tensor.dot(state_below, tparams[_p(prefix, 'W')])+tparams[_p(prefix,'b')]
    result = eval(activ)(result)
    result += state_below
    # input is a 2 or more dimension tensor last dimension is the dim to normalize
    # epsilon is a small value that prevents zero devision 
    # gamma-beta is the vector for the normalization scale and shift
    result=normalize_layer(result, 
                           tparams[_p(prefix, 'gamma')], 
                           tparams[_p(prefix, 'beta')])
    
    return result

# Conditional GRU layer with Attention
def param_init_multiAtt(options, params, prefix='full_att', 
                       dim=None,dim_key=None,dim_value=None, head=None):
    if dim is None: 
        dim = options['dim']
    if head is None:
        head = options['head_num']
    if dim_key is None: 
        dim_key = options['dim']
    if dim_value is None:
        dim_value = dim_key 
    
    # for the multi-attention part(one head)project the input dimension into K,V,Q
    # which is sliced in further use for softmax
    W_key=norm_weight(nin=dim, nout=dim_key)
    params[_p(prefix, 'W_key')]=W_key
    b_key=numpy.zeros((dim_key,)).astype('float32')
    params[_p(prefix, 'b_key')]=b_key
    
    W_value=norm_weight(nin=dim, nout=dim_value)
    params[_p(prefix, 'W_value')]=W_value
    b_value=numpy.zeros((dim_value,)).astype('float32')
    params[_p(prefix, 'b_value')]=b_value
    
    W_query=norm_weight(nin=dim, nout=dim_key)
    params[_p(prefix, 'W_query')]=W_query
    b_query=numpy.zeros((dim_key,)).astype('float32')
    params[_p(prefix, 'b_query')]=b_query
    # for batch normalization
    gamma = numpy.ones((dim,)).astype('float32')
    params[_p(prefix, 'gamma')]=gamma
    beta = numpy.zeros((dim,)).astype('float32')
    params[_p(prefix, 'beta')]=beta
    
    return params

def multiAtt_layer(tparams, options, trng, use_noise,
                  queries, keys, prefix='full_att',
                  queries_mask=None, keys_mask=None,
                  head_num =None, dim_key=None,dim_value=None,
                  epsilon=1e-8, future_block=False,
                  **kwargs):
    # input shape: timestep*nsamples*dim 
    # the second layer of decoder query is the translated target, key is the source vectors
    # when the key=query, here applies self-attention
    assert queries.ndim==3, 'query must be 3D'
    assert keys.ndim==3, 'key must be 3D'
    
    # the key/query mask shape: timesteps*nsamples, the weight'd be softmax along the existing targets
    # when training, the mask is given; translating, the mask is None
    # which there is beam-size sample as a batch
    if queries_mask is None:
        queries_mask=tensor.alloc(1., queries.shape[0],queries.shape[1])
    if keys_mask is None:
        keys_mask = tensor.alloc(1., keys.shape[0], keys.shape[1])

    if head_num is None:
        head_num = options['head_num']
    if dim_key is None:
        dim_key = options['dim']/head_num
    if dim_value is None:
        dim_value = dim_key
    if future_block==True:
        future_block = tensor.alloc(1., 1)
    else:
        future_block = tensor.alloc(0., 1)
    dim=options['dim']   
    timestep_k=keys.shape[0]
    timestep_q=queries.shape[0]
    nsamples=keys.shape[1]
    
    query = tensor.dot(queries,tparams[_p(prefix, 'W_query')])+tparams[_p(prefix, 'b_query')]
    query = tensor.nnet.relu(query)
    key = tensor.dot(keys,tparams[_p(prefix, 'W_key')])+tparams[_p(prefix, 'b_key')]
    key = tensor.nnet.relu(key)
    value = tensor.dot(keys,tparams[_p(prefix, 'W_value')])+tparams[_p(prefix, 'b_value')]
    value = tensor.nnet.relu(value)
    # reshape KQV slice for heads: timestep, head*nsamples, dim/head
    query = concatenate(split(query, [dim/head_num]*head_num,
                        head_num, axis=-1), axis=1)
    key = concatenate(split(key,[dim/head_num]*head_num, 
                        head_num, axis=-1), axis=1)
    value = concatenate(split(value, [dim/head_num]*head_num,
                        head_num, axis=-1), axis=1)
    # result and scale shape: head*nsamples, timestepQ, timestepK
    alpha=tensor.batched_dot(query.dimshuffle([1,0,2]), key.dimshuffle([1,2,0]))/(dim_key**0.5)
    # the key mask is tiled into timestepK, head*nsamples
    alpha*=tensor.tile(keys_mask, [1,head_num]).dimshuffle([1, 'x', 0])
    alpha_filted=tensor.switch(tensor.gt(future_block, 0.),#future_block>0
                               alpha*tensor.tril(tensor.ones((timestep_q,timestep_k)))[None,:,:],
                               alpha
                               )
    # numeric-stable soft-max alpha over the last axis timestepK
    alpha_filted = softmax(alpha_filted)
    # alpha_filted = alpha_filted/alpha_filted.sum(axis=-1,keepdims=True)
    # the query mask 
    alpha_filted *= tensor.tile(queries_mask,[1,head_num]).dimshuffle([1,0,'x'])
    # attention dropout
    if options['use_dropout']:
        alpha_filed=dropout_layer(alpha_filted, use_noise, trng, drop_rate=0.1)
    # get the weight over the value: head*nsamples, timestepQ, dim/head
    result = tensor.batched_dot(alpha_filted,value.dimshuffle([1,0,2]))
    # attention result is nsample, timestep_q, dim
    attention_result= concatenate(split(result, [nsamples]*head_num, head_num, axis=0), 
                                         axis=-1)

    # residual connection and normalize
    attention_result+=queries.dimshuffle([1,0,2])
    # and layer normalize    
    result = normalize_layer(attention_result,
                             tparams[_p(prefix,'gamma')], 
                             tparams[_p(prefix,'beta')]
                             )
    result = result.dimshuffle([1,0,2])
#     mean = tensor.mean(attention_result,axis=-1,keepdims=True)
#     variation=tensor.var(attention_result,axis=-1,keepdims=True)
#     result = (attention_result-mean)/((variation+epsilon)**0.5)
#     result = result*tparams[_p(prefix,'gamma')][None,None,:]+\
#             tparams[_p(prefix,'beta')][None,None,:]
    # result shape: timestep*nsamples(beam)*dim
    return result

def init_params(options):
    params = OrderedDict()
    # embedding
    params['Wemb'] = norm_weight(options['n_words_src'], options['dim'])
    params['Wemb_dec'] = norm_weight(options['n_words'], options['dim'])

    # encoder: 6 layers of multi-Attention with a residual feed forward layer
    for index in numpy.arange(options['layer_num']):
        params = get_layer(options['encoder'])[0](options, params,
                                                  prefix='multiAttEnc{}'.format(index),
                                                  dim=options['dim'])
        # feedforward layer with residual connection and normalization
        params = get_layer('ffNorm')[0](options,params,
                                        prefix='ffNormEnc{}'.format(index),
                                        nin=options['dim'], nout=options['dim'],
                                        ortho=False)
        
    # decoder: 6 layers of multi-attention with a residual feed forward and normalization
    for index in numpy.arange(options['layer_num']):
        params = get_layer(options['decoder'])[0](options, params,
                                                  prefix='maskedMultiAttDec{}'.format(index),
                                                  dim=options['dim'])
        params = get_layer(options['decoder'])[0](options, params,
                                                  prefix='multiAttDec{}'.format(index),
                                                  dim=options['dim'])
        # feedforward layer with residual connection and normalization
        params = get_layer('ffNorm')[0](options,params,
                                        prefix='ffNormDec{}'.format(index),
                                        nin=options['dim'],nout=options['dim'],
                                        ortho=False)
        
    # output feedforward with a softmax layer
    params = get_layer('ff')[0](options, params, prefix='ff_output',
                                nin=options['dim'], nout=options['n_words'],
                                ortho=False)
    return params

# build a training model
def build_model(tparams, options):
    opt_ret = dict()
    # when training, the parallel mask is given. trng is used in dropout
    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: # words*samples
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')
    y = tensor.matrix('y', dtype='int64')
    y_mask = tensor.matrix('y_mask', dtype='float32')

    n_timesteps = x.shape[0]
    n_timesteps_trg = y.shape[0]
    n_samples = x.shape[1]

    # word embedding for forward rnn (source)
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim']])
    # embedding with position embedding
    emb_position=position_emb(emb, options['dim'])
    emb_position=tensor.nnet.sigmoid(emb_position)    
    input_Enc=emb+emb_position
    if options['use_dropout']:
        input_Enc = dropout_layer(input_Enc, use_noise, trng, drop_rate=0.1)
    
    # encoder: 6 layers of attention with normalized feedforward layers
    for index in numpy.arange(options['layer_num']):
        # multi-attention layer
        input_Enc = get_layer(options['encoder'])[1](tparams,options,
                                                trng,use_noise,
                                                 queries=input_Enc,
                                                 keys=input_Enc,
                                                 queries_mask=x_mask,
                                                 keys_mask=x_mask,
                                                 prefix='multiAttEnc{}'.format(index))
        if options['use_dropout']:
            input_Enc=dropout_layer(input_Enc, use_noise, trng, drop_rate=0.1)
        # the normalized feedforward layer:
        input_Enc = get_layer('ffNorm')[1](tparams, options,
                                           state_below=input_Enc,
                                           prefix='ffNormEnc{}'.format(index), )
        if options['use_dropout']:
            input_Enc = dropout_layer(input_Enc, use_noise, trng, drop_rate=0.1)
    
    # shape of src_ctx is timesteps*nsamples*dim
    src_ctx = input_Enc 
    # building decoder the target answer with emb_y shifted right
    emb_y = tparams['Wemb_dec'][y.flatten()]
    emb_y = emb_y.reshape([n_timesteps_trg, n_samples, options['dim']])
    emb_shift_y = tensor.zeros_like(emb_y)
    emb_shift_y = tensor.set_subtensor(emb_shift_y[1:], emb_y[:-1])
    emb_y = emb_shift_y
    # all previous target position embedding
    emb_position_y=position_emb(emb_y, options['dim'])
    emb_position_y=tensor.nnet.sigmoid(emb_position_y)
    input_Dec=emb_y+emb_position_y
    if options['use_dropout']:
        input_Dec=dropout_layer(input_Dec, use_noise, trng, drop_rate=0.1)
        
    # decoder: 6 layers of attention with normalized feedforward layers
    for index in numpy.arange(options['layer_num']):
        # multi-attention layer for the translated target: self attention
        input_Dec = get_layer(options['decoder'])[1](tparams,options,
                                                trng,use_noise,
                                                 queries=input_Dec,
                                                 keys=input_Dec,
                                                 queries_mask=y_mask,
                                                 keys_mask=y_mask,
                                                 future_block=True,
                                                 prefix='maskedMultiAttDec{}'.format(index))
        if options['use_dropout']:
            input_Dec=dropout_layer(input_Dec, use_noise, trng, drop_rate=0.1)
        # multi-attention layer from the source CTX,
        # query is the translated target, key is the source CTX
        input_Dec = get_layer(options['decoder'])[1](tparams,options,
                                                 trng,use_noise,
                                                 queries=input_Dec,
                                                 keys=src_ctx,
                                                 queries_mask=y_mask,
                                                 keys_mask=x_mask,
                                                 prefix='multiAttDec{}'.format(index))
        if options['use_dropout']:
            input_Dec=dropout_layer(input_Dec, use_noise, trng, drop_rate=0.1)
            
        # feedforward layer with normalization
        input_Dec = get_layer('ffNorm')[1](tparams, options, 
                                           state_below=input_Dec, 
                                           prefix='ffNormDec{}'.format(index))
        if options['use_dropout']:
            input_Dec = dropout_layer(input_Dec, use_noise, trng, drop_rate=0.1)
    # the output feedforward(linear) layer with a softmax
    logit = get_layer('ff')[1](tparams,options,
                               state_below=input_Dec,
                               prefix='ff_output',
                               activ='linear')
    logit_shp=logit.shape
    # the result is timestep*nsamples*n_words
    probs=softmax(logit.reshape([logit.shape[0]*logit.shape[1],
                                             logit.shape[2]]))
    # cost
    y_flat = y.flatten()
    y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words']+y_flat
    cost = -tensor.log(probs.flatten()[y_flat_idx])
    cost = cost.reshape([y.shape[0], y.shape[1]])
    cost = (cost*y_mask).sum(0)

    return trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost

# build a sampler
def build_sampler(tparams, options, trng, use_noise):
    # in translation, src_ctx nsamples is 1 by default, x shape: timesteps*nsamples
    # in beam search, the ctx is tiled into beam samples 
    x = tensor.matrix('x', dtype='int64')
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # index 2 word embedding and the position embedding
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim']])
    emb_position=position_emb(emb, options['dim'])
    emb_position=tensor.nnet.sigmoid(emb_position)    
    input_Enc=emb+emb_position 
    if options['use_dropout']:
        input_Enc = dropout_layer(input_Enc, use_noise, trng, drop_rate=0.1)
    
    # encoder: 6 layers of attention with normalized feedforward layers
    for index in numpy.arange(options['layer_num']):
        # multi-attention layer
        input_Enc = get_layer(options['encoder'])[1](tparams,options,
                                                trng,use_noise,
                                                queries=input_Enc,
                                                keys=input_Enc,
                                                prefix='multiAttEnc{}'.format(index))
        if options['use_dropout']:
            input_Enc=dropout_layer(input_Enc, use_noise, trng, drop_rate=0.1)
        # the normalized feedforward layer:
        input_Enc = get_layer('ffNorm')[1](tparams, options, 
                                           state_below=input_Enc, 
                                           prefix='ffNormEnc{}'.format(index))
        if options['use_dropout']:
            input_Enc = dropout_layer(input_Enc, use_noise, trng, drop_rate=0.1)
    
    # shape of src_ctx is timesteps*1*dim
    src_ctx = input_Enc 
    print 'Building f_init...',
    inps = [x]
    outs = [src_ctx]
    f_init = theano.function(inps, outs, name='f_init', profile=profile)
    print 'Done'
    
    # y should hold all of the previous results(timestep*nsamples) 
    y= tensor.matrix('y', dtype='int64')
    trg_timesteps=y.shape[0]
    trg_samples=y.shape[1]
    # if it's the first word(indicate by -1), emb should be all zero and it is indicated by -1
    # index2embedding: timestep*nsamples->timestep*nsamples*dim 
    y_index=y.flatten()# 1D timestep*nsamples,
    emb_y = tensor.switch(y_index[:, None] < 0,
                        tensor.alloc(0., 1, options['dim']),
                        tparams['Wemb_dec'][y_index] )
    emb_y = emb_y.reshape([trg_timesteps,trg_samples,options['dim']])
    # all previous results' embedding, reshape: timestep*nsamples*dim
    emb_position_y=position_emb(emb_y, options['dim'])
    emb_position_y=tensor.nnet.sigmoid(emb_position_y)
    input_Dec = emb_y + emb_position_y
    if options['use_dropout']:
        input_Dec = dropout_layer(input_Dec, use_noise, trng, drop_rate=0.1)
    
    # decoder: 6 layers of attention with normalized feedforward layers
    for index in numpy.arange(options['layer_num']):
        # multi-attention layer for the translated target: self attention
        input_Dec = get_layer(options['decoder'])[1](tparams,options,
                                                trng,use_noise,
                                                queries=input_Dec,
                                                keys=input_Dec,
                                                future_block=True,
                                                prefix='maskedMultiAttDec{}'.format(index))
        if options['use_dropout']:
            input_Dec=dropout_layer(input_Dec, use_noise, trng, drop_rate=0.1)
        # multi-attention layer from the source CTX,
        # query is the translated target, key is the source CTX
        input_Dec = get_layer(options['decoder'])[1](tparams,options,
                                                 trng,use_noise,
                                                 queries=input_Dec,
                                                 keys=src_ctx,
                                                 prefix='multiAttDec{}'.format(index))
        if options['use_dropout']:
            input_Dec=dropout_layer(input_Dec, use_noise, trng, drop_rate=0.1)
        # feedforward layer with normalization
        input_Dec = get_layer('ffNorm')[1](tparams, options, 
                                           state_below=input_Dec, 
                                           prefix='ffNormDec{}'.format(index))
        if options['use_dropout']:
            input_Dec = dropout_layer(input_Dec, use_noise, trng, drop_rate=0.1)
    
    # shape of logit: timestep*beam*dim
    logit = get_layer('ff')[1](tparams,options,
                               state_below=input_Dec,
                               prefix='ff_output',
                               activ='linear')
    # only take the last timestep(squeezed) as the current output:
    logit = logit[-1,:,:].reshape([logit.shape[1],logit.shape[2]])
    next_probs = softmax(logit)

    # sample from softmax distribution to get the sample(nsamples*trg_words)
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)
    
    # compile a function to do the whole thing above, next word probability,
    # sampled word for the next target, next hidden state to be used
    print 'Building f_next...',
    inps = [y,src_ctx] # 
    outs = [next_probs, next_sample]# 
    f_next = theano.function(inps, outs, name='f_next', profile=profile)
    print 'Done'
    return f_init, f_next

# generate sample, either with stochastic sampling or beam search. Note that,
# this function iteratively calls f_init and f_next functions.
def gen_sample(tparams, f_init, f_next, x, options, trng=None, k=1, maxlen=30,
               stochastic=True, argmax=False, ):
    # k is the beam size we have
    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'
    sample = []
    sample_score = []
      
    if stochastic:
        sample_score = 0
    # beam search: counter
    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k        
    hyp_scores = numpy.zeros(live_k).astype('float32')

    # get initial state of decoder rnn and encoder context
    ret = f_init(x)
    src0 = ret[0]
    # bos indicator: -1 initiated with timestep*nsamples
    result_history = -1 * numpy.ones((1,1)).astype('int64')  

    for ii in xrange(maxlen):
        # expand the src into beam samples 
        src = numpy.tile(src0,[live_k,1]) 
        inps = [result_history, src]
        ret = f_next(*inps)
        next_p, next_w = ret[0], ret[1]

        if stochastic: # beam=1 with multinomial sampling
            if argmax:# direct argmax 
                nw = next_p[0].argmax()
            else:# argmax from multinomial sampling
                nw = next_w[0]
            sample.append(nw)
            sample_score -= numpy.log(next_p[0, nw])
            # update result history for the next prediction: timestep*nsample
            nwList=numpy.expand_dims([nw], axis=1)
            result_history=numpy.concatenate([result_history, nwList ])
            if nw == 0:
                break
        else:
            # beam search: cand_score shape=beam * word_trg
            cand_scores = hyp_scores[:, None] - numpy.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)]
            # top-beam result in beam*word_trg matrix
            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros((k-dead_k)).astype('float32')

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                # record top-beam word index
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            
            for idx in xrange(len(new_hyp_samples)):
                # for every index in record, clear the finished sample
                if new_hyp_samples[idx][-1] == 0:# the most likely result=EOS
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    # this is a newly allocated history for further search
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
            
            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break
            # update the history for next decoding step: nsamples*timestep(need transpose)
            # the samples musts be append with -1 as start
            # the result_history must be resembled into 
            bos= -1*numpy.ones((len(hyp_samples) ,1)).astype('int64')
            result_history = numpy.concatenate([bos, 
                                                numpy.array(hyp_samples)],
                                                axis=1).T

    if not stochastic:
        # dump the rest 
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])   
    
    return sample, sample_score

# calculate the log probablities on a given corpus using translation model
def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=True):
    probs = []
    n_done = 0
    for x, y in iterator:
        n_done += len(x)

        x, x_mask, y, y_mask = prepare_data(x, y,
                                            n_words_src=options['n_words_src'],
                                            n_words=options['n_words'])

        pprobs = f_log_probs(x, x_mask, y, y_mask)
        for pp in pprobs:
            probs.append(pp)

        if numpy.isnan(numpy.mean(probs)):
            ipdb.set_trace()

        if verbose:
            print >>sys.stderr, '%d samples computed' % (n_done)

    return numpy.array(probs)


# optimizers
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adam(lr, tparams, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup, profile=profile)

    lr0 = 0.0002
    b1 = 0.1
    b2 = 0.02
    e = 1e-9
    # for dynamic lrate
    warmup_step=4000
    dim=512 # dim of the model
    
    updates = []
    # lrate is dynamic: increased dynamically during warmup steps 
    # and decreased proportionally to step number
    i = theano.shared(numpy.float32(0.))
    i_t = i + 1.
    # fix1 = 1. - b1**(i_t)
    # fix2 = 1. - b2**(i_t)
    # lr_t = lr0 * (tensor.sqrt(fix2) / fix1)
    lr_t = dim**(-0.5)* tensor.minimum(i_t**(-0.5),i_t*warmup_step**(-1.5))

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([lr], [], updates=updates,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update

def adadelta(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rg2up,
                                    profile=profile)

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads, running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update

def rmsprop(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rgup+rg2up,
                                    profile=profile)

    updir = [theano.shared(p.get_value() * numpy.float32(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update

def sgd(lr, tparams, grads, x, mask, y, cost):
    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    profile=profile)

    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]
    f_update = theano.function([lr], [], updates=pup, profile=profile)

    return f_grad_shared, f_update

def train(dim=1000,  # model dimension
          encoder='fullAtt',
          decoder='fullAtt',
          layer_num=2,
          head_num=8,
          patience=10,  # early stopping patience
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,  # L2 regularization penalty
          alpha_c=0.,  # alignment regularization
          clip_c=-1.,  # gradient clipping threshold
          warmup_step=4000,# warmup learning rate for adam
          lrate=0.01,  # learning rate
          n_words_src=100000,  # source vocabulary size
          n_words=100000,  # target vocabulary size
          maxlen=100,  # maximum length of the description
          optimizer='adam',
          batch_size=500,
          valid_batch_size=500,
          saveto='model.npz',
          validFreq=1000,
          saveFreq=1000,   # save the parameters after every saveFreq updates
          sampleFreq=100,   # generate some samples after every sampleFreq
          datasets=[
              '/data/cn.tok',
              '/data/en.tok'],
          valid_datasets=['../data/MT02.cn.dev',
                          '../data/MT02.en.dev'],
          dictionaries=[
              '/data/cn.tok.pkl',
              '/data/en.tok.pkl'],
          use_dropout=False,
          reload_=False,
          overwrite=False,
          **bleu_params
          ):

    # Model options
    model_options = locals().copy()
    # BLEU validation
    bleu_valid = BleuValidator(model_options,**bleu_params)

    # load dictionaries and invert them
    worddicts = [None] * len(dictionaries)
    worddicts_r = [None] * len(dictionaries)
    for ii, dd in enumerate(dictionaries):
        with open(dd, 'rb') as f:
            worddicts[ii] = pkl.load(f)
        worddicts_r[ii] = dict()
        for kk, vv in worddicts[ii].iteritems():
            worddicts_r[ii][vv] = kk

    # reload options
    if reload_ and os.path.exists(saveto):
        print 'Reloading model options'
        with open('%s.pkl' % saveto, 'rb') as f:
            model_options = pkl.load(f)

    print 'Loading data'
    train = TextIterator(datasets[0], datasets[1],
                         dictionaries[0], dictionaries[1],
                         n_words_source=n_words_src, n_words_target=n_words,
                         batch_size=batch_size,
                         maxlen=maxlen)
    valid = TextIterator(valid_datasets[0], valid_datasets[1],
                         dictionaries[0], dictionaries[1],
                         n_words_source=n_words_src, n_words_target=n_words,
                         batch_size=valid_batch_size,
                         maxlen=maxlen)

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        print 'Reloading model parameters'
        params = load_params(saveto, params)

    tparams = init_tparams(params)

    trng, use_noise, \
        x, x_mask, y, y_mask, \
        opt_ret, \
        cost = \
        build_model(tparams, model_options)
    inps = [x, x_mask, y, y_mask]

    print 'Building sampler'
    f_init, f_next = build_sampler(tparams, model_options, trng, use_noise)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    print 'Done'
    cost = cost.mean()

    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # regularize the alpha weights
    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * (
            (tensor.cast(y_mask.sum(0)//x_mask.sum(0), 'float32')[:, None] -
             opt_ret['dec_alphas'].sum(0))**2).sum(1).mean()
        cost += alpha_reg

    # after all regularizers - compile the computational graph for cost
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=profile)
    print 'Done'

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'

    # apply gradient clipping here
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)
    print 'Done'

    print 'Optimization'

    best_p = None
    bad_counter = 0
    uidx = 0
    estop = False
    history_errs = []
    history_bleu = []
    valid_not_fin = []
    # reload history
    if reload_ and os.path.exists(saveto):
        rmodel = numpy.load(saveto)
        if 'history_errs' in rmodel:
            history_errs = list(rmodel['history_errs'])
        if 'history_bleu' in rmodel:
            history_bleu = list(rmodel['history_bleu'])
        if 'uidx' in rmodel:
            uidx = rmodel['uidx']

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size

    for eidx in xrange(max_epochs):
        n_samples = 0

        for x, y in train:
            n_samples += len(x)
            uidx += 1 # updating time
            use_noise.set_value(1.)

            x, x_mask, y, y_mask = prepare_data(x, y, maxlen=maxlen,
                                                n_words_src=n_words_src,
                                                n_words=n_words)

            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue
            ud_start = time.time()

            # compute cost, grads and copy grads to shared variables
            cost = f_grad_shared(x, x_mask, y, y_mask)
            # do the update on parameters
            f_update(lrate)
            ud = time.time() - ud_start

            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.
            # verbose / update info display
            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud
            # save the best model so far, in addition, save the latest model
            # into a separate file with the iteration number for external eval

            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving the best model...',
                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                numpy.savez(saveto, 
                            history_errs=history_errs, 
                            history_bleu=history_bleu,
                            uidx=uidx, **params)
                pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'))
                print 'Done'                    
                # save with every uidx
                if not overwrite:
                    print 'Saving the model at iteration {}...'.format(uidx),
                    saveto_uidx = '{}.iter{}.npz'.format(
                        os.path.splitext(saveto)[0], uidx)
                    numpy.savez(saveto_uidx, 
                                history_errs=history_errs,
                                history_bleu=history_bleu,
                                uidx=uidx, **unzip(tparams))
                    print 'Done'

            # generate some samples with the model and display them
            if numpy.mod(uidx, sampleFreq) == 0:
                # FIXME: random selection?
                for jj in xrange(numpy.minimum(5, x.shape[1])):
                    stochastic = True
                    sampleData = gen_sample(tparams, f_init, f_next,
                                           x[:, jj][:, None],
                                           model_options, trng=trng, k=1,
                                           maxlen=30,
                                           stochastic=stochastic,
                                           argmax=False)
                    sample = sampleData[0]
                    score = sampleData[1]
                    print 'Source ', jj, ': ',
                    for vv in x[:, jj]:
                        if vv == 0:
                            break
                        if vv in worddicts_r[0]:
                            print worddicts_r[0][vv],
                        else:
                            print 'UNK',
                    print
                    print 'Truth ', jj, ' : ',
                    for vv in y[:, jj]:
                        if vv == 0:
                            break
                        if vv in worddicts_r[1]:
                            print worddicts_r[1][vv],
                        else:
                            print 'UNK',
                    print
                    print 'Sample ', jj, ': ',
                    if stochastic:
                        ss = sample
                    else:
                        score = score / numpy.array([len(s) for s in sample])
                        ss = sample[score.argmin()]
                    for vv in ss:
                        if vv == 0:
                            break
                        if vv in worddicts_r[1]:
                            print worddicts_r[1][vv],
                        else:
                            print 'UNK',
                    print
###################################################################################
            # validate model on validation set and early stop if necessary
            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                valid_errs = pred_probs(f_log_probs, prepare_data,
                                        model_options, valid)
                valid_err = valid_errs.mean()
                history_errs.append(valid_err)
            # bleu validation only happends when valid cost came below 100
            # or the bleu is always 0 then early stops when training starts
                if numpy.mod(uidx,validFreq*10)==0 and valid_err<75:
                    # save a independent model for bleu validation:
                    temp_p = unzip(tparams)
                    print 'Saving (Check Point) at iteration {}...'.format(uidx),
                    saveto_uidx = '{}.iter{}.npz'.format(
                        os.path.splitext(saveto)[0], uidx)
                    numpy.savez(saveto_uidx, 
                                history_errs=history_errs,
                                history_bleu=history_bleu,
                                uidx=uidx, **unzip(tparams))
                    print 'Done'
                    # validate the saved model and get bleu , subprocess 
                    # is stored and checked before a new validation
                    if len(valid_not_fin) > 0:
                        popen,previous_p,previous_model,previous_trans,previous_uidx,previous_eidx = \
                            valid_not_fin.pop()
                        popen.wait()
                        valid_bleu = bleu_valid.testBLEU(previous_trans)
                        history_bleu.append(valid_bleu)
                        print 'Epoch %d Update %d BLEU %f\t Best BLEU %f' %\
                            (eidx, uidx, valid_bleu, numpy.array(history_bleu).max())
                        # remove the temporal model used in validation
                        if numpy.mod(previous_uidx,validFreq*100)!=0:
                            print 'remove temp file... '+previous_model+' '+previous_trans ,
                            bleu_valid.remove_temp_file(model_file=previous_model,trans_file=previous_trans)                        
                            print ' Done'
                        if previous_uidx == 0 or valid_bleu>=numpy.array(history_bleu).max():
                            best_p = previous_p
                            bad_counter_bleu = 0
                        if len(history_bleu) > patience_bleu and valid_bleu <=\
                                numpy.array(history_bleu)[:-patience_bleu].max():
                            bad_counter_bleu += 1
                            if bad_counter_bleu > patience_bleu:
                                print 'Early Stop'
                                estop = True
                                break
                        
                    trans_saveto="transMem.iter%d" % uidx
                    popen = bleu_valid.decode(theano.config.device,
                                              trans_saveto,
                                              saveto_uidx,
                                              )
                    valid_not_fin.append((popen,temp_p,saveto_uidx,trans_saveto,copy.deepcopy(uidx),eidx))
                else:
                # using default validation with valid_errs
                    if uidx == 0 or valid_err <= numpy.array(history_errs).min():
#                         best_p = unzip(tparams)
                        bad_counter = 0
                    if len(history_errs) > patience and valid_err >= \
                            numpy.array(history_errs)[:-patience].min():
                        bad_counter += 1
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break  

                if numpy.isnan(valid_err):
                    ipdb.set_trace()

                print 'Valid ', valid_err
###############################################################################
            # finish after this many updates
            if uidx >= finish_after:
                print 'Finishing after %d iterations!' % uidx
                estop = True
                break

        print 'Seen %d samples' % n_samples

        if estop:
            break
        
    if len(valid_not_fin)>0:
        popen,previous_p,previous_model,previous_trans,previous_uidx,previous_eidx = \
                valid_not_fin.pop()
        popen.wait()
        valid_bleu = bleu_valid.testBLEU(previous_trans)
        history_bleu.append(valid_bleu)
        if valid_bleu > numpy.array(history_bleu).max():
            best_p=previous_p
            os.system('cp %s %s' % (previous_model,saveto))
        print 'Final Update %d BLEU %f\t Best BLEU %f' \
              % (uidx, this_bleu, numpy.array(history_bleu).max())

    if best_p is not None:
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    valid_err = pred_probs(f_log_probs, prepare_data,
                           model_options, valid).mean()

    print 'Valid ', valid_err

    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p,
                history_errs=history_errs,
                history_bleu=history_bleu,
                uidx=uidx,**params)

    return valid_err


if __name__ == '__main__':
    pass
