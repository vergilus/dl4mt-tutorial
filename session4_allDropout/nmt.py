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
# import pydot

from collections import OrderedDict
from bleu_validator import BleuValidator
from data_iterator import TextIterator

profile = False


# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]

# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=0.5, n=1,
                                     dtype=state_before.dtype),
        state_before * 0.5)
    return proj

def shared_dropout_layer(shape, use_noise, trng, retain_rate=1):
    if retain_rate == 1:
        proj = tensor.ones(shape,dtype='float32')
    else:
        proj = tensor.switch(
            use_noise, 
            trng.binomial(shape,p=retain_rate,n=1,dtype='float32'), 
            theano.shared(numpy.float32(retain_rate))) 
    return proj

# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]

    return params

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'buffer_init':('param_init_buffer','buffer_init_layer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'memDec': ('param_init_memDec', 'memDec_layer'),
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

def norm_bias(size, dim,scale=None):
    if scale is None:
        scale=1
    bias=scale * numpy.random.normal(0.0, 0.1, (size,dim))
    return bias.astype('float32')

def tanh(x):
    return tensor.tanh(x)

def sigmoid(x):
    return tensor.nnet.sigmoid(x)

def linear(x):
    return x


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


# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None,
                       ortho=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(
        tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
        tparams[_p(prefix, 'b')])

def param_init_buffer(options, params, prefix='init_buffer',size=None,dim=None,nin=None, 
                      ):
    if nin is None:
        nin=options['dim']
    if size is None:
        size = options['buffer_size']
    if dim is None:
        dim =options['buffer_dim']
        
    params[_p(prefix, 'W')] = norm_weight(nin, dim, scale=0.01)
    params[_p(prefix, 'v')] = norm_bias(size,dim)

    return params

def buffer_init_layer(tparams,state_below,options,
                      timestep=None,mask=None,
                      prefix='init_buffer',
                      activ = 'lambda x : tensor.nnet.tanh(x)',**kwargs):  
      
    assert state_below.ndim == 2 ,'context_sum for initiation must be 2D'
#     print 'initiating buffer'
    if mask is None:
        # sampler of one-step encoder initiation: 
        assert timestep ,'when mask is None, timestep must be provided!'
        mask = tensor.alloc(1., timestep, 1)
        buffer =( ( eval(activ)(
                tensor.dot(state_below,tparams[_p(prefix,'W')])
                ) )/mask.sum(0)[:, None] )[None,:,:] + tparams[_p(prefix,'v')][:,None,:]   

    else:
        buffer =( ( eval(activ)(
                tensor.dot(state_below,tparams[_p(prefix,'W')])
                ) )/mask.sum(0)[:, None] )[None,:,:] + tparams[_p(prefix,'v')][:,None,:]   
    return buffer 

# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    # embedding to gates transformation weights, biases
    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')

    # recurrent transformation weights for gates
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U

    # embedding to hidden state proposal weights, biases
    Wx = norm_weight(nin, dim)
    params[_p(prefix, 'Wx')] = Wx
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim)
    params[_p(prefix, 'Ux')] = Ux

    return params


def gru_layer(tparams, state_below, options, prefix='gru', mask=None,
              emb_dropout=None,
              rec_dropout=None,
              **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # state_below is the input word embeddings
    # input to the gates, concatenated
    state_below_ = tensor.dot(state_below*emb_dropout[0], tparams[_p(prefix, 'W')]) + \
        tparams[_p(prefix, 'b')]
    # input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below*emb_dropout[1], tparams[_p(prefix, 'Wx')]) + \
        tparams[_p(prefix, 'bx')]

    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(m_, x_, xx_, h_, U, Ux, rec_dropout):
        preact = tensor.dot(h_* rec_dropout[0], U)
        preact += x_

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # compute the hidden state proposal
        preactx = tensor.dot(h_*rec_dropout[1], Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        # hidden state proposal
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]
    init_states = [tensor.alloc(0., n_samples, dim)]
    _step = _step_slice
    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')],
                   rec_dropout,
                   ]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_states,
                                non_sequences=shared_vars,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)
    rval = [rval]
    return rval


# Conditional GRU layer with Attention
def param_init_memDec(options, params, prefix='memDec',
                        nin=None, dim=None, dimctx=None,
                        nin_nonlin=None, dim_nonlin=None,
                        buffer_size=None,buffer_dim=None,):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']        
    if nin_nonlin is None: #hidden state input dimension
        nin_nonlin = nin
    if dim_nonlin is None: # hidden state dimension
        dim_nonlin = dim
    if buffer_size is None:
        buffer_size = options['buffer_size']
    if buffer_dim is None:
        buffer_dim = options['buffer_dim']

    # attention: context projection
    Wc_att = norm_weight(dimctx)
    params[_p(prefix, 'Wc_att')] = Wc_att

    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix, 'b_att')] = b_att
    
    #BI-RNN embedding projection for previous hidden state 
    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')
    
    U = numpy.concatenate([ortho_weight(dim_nonlin),
                           ortho_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'U')] = U

    Wx = norm_weight(nin_nonlin, dim_nonlin)
    params[_p(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim_nonlin)
    params[_p(prefix, 'Ux')] = Ux
    params[_p(prefix, 'bx')] = numpy.zeros((dim_nonlin,)).astype('float32')
    #buffer projection 
    Wc_buf = norm_weight(buffer_dim)
    params[_p(prefix, 'Wc_buf')]= Wc_buf
    
    b_buf = numpy.zeros((buffer_dim,)).astype('float32')
    params[_p(prefix, 'b_buf')] = b_buf

    # calculate read/write weight for buffer memory accessing    
    W_buf_read = norm_weight(dim , buffer_dim)
    params[_p(prefix, 'W_buf_read')]=W_buf_read
    
    U_buf_read = norm_weight(buffer_dim, 1)
    params[_p(prefix, 'U_buf_read')] = U_buf_read
    
    c_buf_read = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_buf_read')] = c_buf_read
    # reading interpolation gate and bias
    buf_gate = norm_weight(dim_nonlin, buffer_size)
    params[_p(prefix, 'buf_gate')] = buf_gate 
    
    buf_gate_b = numpy.zeros((buffer_size,)).astype('float32')
    params[_p(prefix, 'buf_gate_b')] = buf_gate_b
    
    # intermediate state calculation
    W_temp_state = norm_weight(buffer_dim,dim_nonlin)
    params[_p(prefix, 'W_temp_state')] = W_temp_state 
    
    U_temp_state = ortho_weight(dim_nonlin)
    params[_p(prefix, 'U_temp_state')] = U_temp_state
    
    b_temp_state = numpy.zeros((dim_nonlin,)).astype('float32')
    params[_p(prefix, 'b_temp_state')] = b_temp_state
    
    # attention: source memory reading
    W_src_read = norm_weight(dim_nonlin, dimctx)
    params[_p(prefix, 'W_src_read')] = W_src_read
    
    U_src_read = norm_weight(dimctx, 1)
    params[_p(prefix, 'U_src_read')] = U_src_read
    
    c_src_read = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_src_read')] = c_src_read
        
    #updating present hidden state
    U_nl = numpy.concatenate([ortho_weight(dim_nonlin),
                              ortho_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'U_nl')] = U_nl   
    params[_p(prefix, 'b_nl')] = numpy.zeros((2 * dim_nonlin,)).astype('float32')
    
    Wc = norm_weight(dimctx, dim*2)
    params[_p(prefix, 'Wc')] = Wc
    
    Ux_nl = ortho_weight(dim_nonlin)
    params[_p(prefix, 'Ux_nl')] = Ux_nl  
    params[_p(prefix, 'bx_nl')] = numpy.zeros((dim_nonlin,)).astype('float32')

    Wcx = norm_weight(dimctx, dim)
    params[_p(prefix, 'Wcx')] = Wcx

    #calculate add and erase
    W_era = norm_weight(dim_nonlin , buffer_dim)
    params[_p(prefix,'W_era')]= W_era
    b_era = numpy.zeros((buffer_dim,)).astype('float32')
    params[_p(prefix,'b_era')]= b_era
      
    W_add = norm_weight(dim_nonlin, buffer_dim)
    params[_p(prefix,'W_add')]= W_add
    b_add = numpy.zeros((buffer_dim,)).astype('float32')
    params[_p(prefix,'b_add')]= b_add


    return params

def memDec_layer(tparams, state_below, options, prefix='memDec',
                   mask=None, context=None, one_step=False,
                   init_state=None, init_memory=None,init_read=None,
                   context_mask=None,
                   emb_dropout=None,ctx_dropout=None,
                   rec_dropout=None,buf_dropout=None,
                   **kwargs):

    assert context, 'Context must be provided'
    if one_step:
        assert init_state, 'previous state must be provided'
        assert init_memory, 'previous buffer must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1] # for build model
    else:
        n_samples = 1 #for build sample

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1) #for build sample

    dim = tparams[_p(prefix, 'Wcx')].shape[1]
    buffer_size=options['buffer_size']
    buffer_dim=options['buffer_dim']
    # initial state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)
    # initial buffer reading weight 
    if init_read is None:
        init_read = tensor.alloc(0., n_samples, buffer_size)

    assert init_memory.ndim==3, \
        'Buffer must be 3-d: #buffer_size x #sample x #buffer_dim'
    # projected context
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x #dim'
    pctx_ = tensor.dot(context*ctx_dropout[0], tparams[_p(prefix, 'Wc_att')]) +\
        tparams[_p(prefix, 'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x for gate and GRU hidden state
    state_belowx = tensor.dot(state_below*emb_dropout[0], tparams[_p(prefix, 'Wx')]) +\
        tparams[_p(prefix, 'bx')]
    state_below_ = tensor.dot(state_below*emb_dropout[1], tparams[_p(prefix, 'W')]) +\
        tparams[_p(prefix, 'b')]

    def _step_slice(m_, x_, xx_, 
                    h_, ctx_, alpha_, buffer_, alpha_buf_,
                    pctx_, cc_,
                    rec_dropout,ctx_dropout,buf_dropout,
                    U,Ux, # previous hidden state
                    Wc_buf,b_buf,# buffer projection
                    W_buf_read, U_buf_read,c_buf_read,# buffer reading
                    buf_gate, buf_gate_b, # reading interpolation
                    W_temp_state,U_temp_state ,b_temp_state,# intermediate hidden state
                    W_src_read, U_src_read, c_src_read,# source reading(context)                        
                    U_nl, b_nl, Wc, Ux_nl, bx_nl, Wcx ,# updating present hidden state
                    W_era,b_era,W_add,b_add, # buffer update
                    ):
        # GRU RNN state 
        preact1 = tensor.dot(h_*rec_dropout[0], U)
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, dim)
        u1 = _slice(preact1, 1, dim)

        preactx1 = tensor.dot(h_*rec_dropout[1], Ux)
        preactx1 *= r1
        preactx1 += xx_

        h1 = tensor.tanh(preactx1)

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_
        # buffer projection 
        pbuf_ = tensor.dot(buffer_*buf_dropout[0], Wc_buf) + b_buf
        # calculate buffer-state match with buffer memory for reading 
        pstate1_ = tensor.dot(h1*rec_dropout[2], W_buf_read)
        pbuf__ = pbuf_ + pstate1_[None, :, :]
        # calculate reading weight for buffer memory
        pbuf__ = tensor.tanh(pbuf__) 
        alpha_buf = tensor.dot(pbuf__*buf_dropout[1], U_buf_read)+c_buf_read
        alpha_buf = alpha_buf.reshape([alpha_buf.shape[0],alpha_buf.shape[1]])
        
        alpha_buf = tensor.nnet.sigmoid(alpha_buf)

        # reading weight interpolation(need transpose)
        gate = tensor.dot(h1*rec_dropout[3], buf_gate) + buf_gate_b
        gate = tensor.nnet.sigmoid(gate)
        alpha_buf = (gate * alpha_buf_).T + (1. - gate.T) * alpha_buf
        
        alpha_buf = alpha_buf/alpha_buf.sum(0,keepdims=True)# normalize
        buf_ = (buffer_ * alpha_buf[:, :, None]).sum(0)# buffer reading result
        
        #calculate intermediate state vector using read result and embedding
        h2 = tensor.dot(buf_*buf_dropout[2], W_temp_state) + b_temp_state
        h2 += tensor.dot(h1*rec_dropout[4], U_temp_state)
        h2 = tensor.tanh(h2)
        
        # calculate source memory(context)-h2 match function
        pstate2_ = tensor.dot(h2*rec_dropout[5],W_src_read)
        pctx__ = pctx_ + pstate2_[None, :, :]
        #pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__*ctx_dropout[1], U_src_read)+c_src_read
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        if context_mask:
            alpha = alpha * context_mask
#         alpha = alpha/alpha.sum(0, keepdims=True)
        alpha = alpha-alpha.max(axis=0, keepdims=True)
        alpha = alpha- tensor.log(tensor.exp(alpha).sum(axis=0,keepdims=True))
        alpha = tensor.exp(alpha)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context
        
        # GRU combining context(ctx_, state) embedding is included in state 
        preact2 = tensor.dot(h2*rec_dropout[6], U_nl)+b_nl
        preact2 += tensor.dot(ctx_*ctx_dropout[2], Wc)
        preact2 = tensor.nnet.sigmoid(preact2)

        r3 = _slice(preact2, 0, dim)
        u3 = _slice(preact2, 1, dim)

        preactx2 = tensor.dot(h2*rec_dropout[7], Ux_nl)+bx_nl
        preactx2 *= r3
        preactx2 += tensor.dot(ctx_*ctx_dropout[3], Wcx)

        h3 = tensor.tanh(preactx2)

        h3 = u3 * h1 + (1. - u3) * h3
        h3 = m_[:, None] * h3 + (1. - m_)[:, None] * h1

        # calculate erase gate 
        erase = tensor.dot(h3*rec_dropout[8],W_era)+b_era
        erase = tensor.nnet.sigmoid(erase)
        # calculate add gate
        add = tensor.dot(h3*rec_dropout[9],W_add)+b_add
        add = tensor.nnet.sigmoid(add)
        
        # update buffer memory(write)
        eraseStuff = alpha_buf[:,:,None] * erase[None,:,:]
        buffer =  (1.-eraseStuff) * buffer_
        addStuff = alpha_buf[:,:,None] * add[None,:,:]
        buffer +=addStuff 

#         buffer=buffer_ # only a writing test
        return h3, ctx_, alpha.T, buffer, alpha_buf.T # pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx]
    #seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [
                   tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'Wc_buf')],
                   tparams[_p(prefix, 'b_buf')],
                   tparams[_p(prefix, 'W_buf_read')],
                   tparams[_p(prefix, 'U_buf_read')],
                   tparams[_p(prefix, 'c_buf_read')],
                   tparams[_p(prefix, 'buf_gate')], 
                   tparams[_p(prefix, 'buf_gate_b')], 
                   tparams[_p(prefix, 'W_temp_state')],
                   tparams[_p(prefix, 'U_temp_state')],
                   tparams[_p(prefix, 'b_temp_state')],
                   tparams[_p(prefix, 'W_src_read')],
                   tparams[_p(prefix, 'U_src_read')],
                   tparams[_p(prefix, 'c_src_read')],
                   tparams[_p(prefix, 'U_nl')],
                   tparams[_p(prefix, 'b_nl')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix, 'Ux_nl')],                 
                   tparams[_p(prefix, 'bx_nl')],
                   tparams[_p(prefix, 'Wcx')],
                   tparams[_p(prefix, 'W_era')],
                   tparams[_p(prefix, 'b_era')],
                   tparams[_p(prefix, 'W_add')],
                   tparams[_p(prefix, 'b_add')],
                   ]

    if one_step: # building sampler
        rval = _step(*(seqs + [init_state, 
                               None, # context result
                               None, # context alpha
                               init_memory, # buffer
                               init_read,  # buffer reading alpha
                               pctx_, context,
                               rec_dropout,ctx_dropout,buf_dropout] +
                       shared_vars))
    else: # building model
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[2]),
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[0]),
                                                  init_memory,
                                                  init_read,
                                                  ],
                                    non_sequences=[pctx_,context,rec_dropout,ctx_dropout,buf_dropout]+shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval

# initialize all parameters
def init_params(options):
    params = OrderedDict()

    # embedding
    params['Wemb'] = norm_weight(options['n_words_src'], options['dim_word'])
    params['Wemb_dec'] = norm_weight(options['n_words'], options['dim_word'])

    # encoder: bidirectional RNN
    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'])
    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder_r',
                                              nin=options['dim_word'],
                                              dim=options['dim'])
    ctxdim = 2 * options['dim']

    # init_state, init_cell
    params = get_layer('ff')[0](options, params, prefix='ff_state',
                                nin=ctxdim, nout=options['dim'])
    
    # buffer_init,buffer_cell
    params = get_layer('buffer_init')[0](options,params,prefix='init_buffer',
                                nin=ctxdim,
                                )
    
    # decoder
    params = get_layer(options['decoder'])[0](options, params,
                                              prefix='decoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'],
                                              dimctx=ctxdim)
    # readout
    params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm',
                                nin=options['dim'], nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_prev',
                                nin=options['dim_word'],
                                nout=options['dim_word'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_ctx',
                                nin=ctxdim, nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit',
                                nin=options['dim_word'],
                                nout=options['n_words'])

    return params


# build a training model
def build_model(tparams, options):
    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')
    y = tensor.matrix('y', dtype='int64')
    y_mask = tensor.matrix('y_mask', dtype='float32')

    # for the backward rnn, we just need to invert x and x_mask
    xr = x[::-1]
    xr_mask = x_mask[::-1]

    n_timesteps = x.shape[0]
    n_timesteps_trg = y.shape[0]
    n_samples = x.shape[1]

    if options['use_dropout']:
        retain_emb=1-options['dropout_emb']
        retain_src=1-options['dropout_src']
        retain_trg=1-options['dropout_trg']
        retain_hidden=1-options['dropout_hidden']
        
        # sentence to embedding
        src_dropout = shared_dropout_layer((n_timesteps,n_samples,1), use_noise, trng, retain_src)
        trg_dropout = shared_dropout_layer((n_timesteps_trg,n_samples,1), use_noise, trng, retain_trg)
        src_dropout = tensor.tile(src_dropout,(1,1,options['dim_word']))
        trg_dropout = tensor.tile(trg_dropout,(1,1,options['dim_word']))
        #in GRU,2 mask is needed(gate and state).BiRNN encoder is 2 GRU
        rec_dropout = shared_dropout_layer((2,n_samples,options['dim']), use_noise, trng, retain_hidden)
        emb_dropout = shared_dropout_layer((2,n_samples,options['dim_word']), use_noise, trng, retain_emb)
        rec_dropout_r = shared_dropout_layer((2,n_samples,options['dim']), use_noise, trng, retain_hidden)
        emb_dropout_r = shared_dropout_layer((2,n_samples,options['dim_word']), use_noise, trng, retain_emb)
        #in decoder(memDec),there are 2 GRU layers,and ctx is used in hidden state update and result 
        rec_dropout_d = shared_dropout_layer((10,n_samples,options['dim']), use_noise, trng, retain_hidden)
        emb_dropout_d = shared_dropout_layer((2,n_samples,options['dim_word']), use_noise, trng, retain_emb)
        ctx_dropout_d = shared_dropout_layer((4,n_samples,2*options['dim']), use_noise, trng, retain_hidden)
        buf_dropout_d = shared_dropout_layer((3,n_samples,options['buffer_dim']),use_noise,trng, retain_hidden)
    else:
        rec_dropout = theano.shared(numpy.array([1.]*2,dtype='float32'))
        emb_dropout = theano.shared(numpy.array([1.]*2,dtype='float32'))
        rec_dropout_r = theano.shared(numpy.array([1.]*2,dtype='float32'))
        emb_dropout_r = theano.shared(numpy.array([1.]*2,dtype='float32'))
        rec_dropout_d = theano.shared(numpy.array([1.]*10,dtype='float32'))
        emb_dropout_d = theano.shared(numpy.array([1.]*2,dtype='float32'))
        ctx_dropout_d = theano.shared(numpy.array([1.]*4,dtype='float32'))
        buf_dropout_d = theano.shared(numpy.array([1.]*3,dtype='float32'))
        
    # word embedding for forward rnn (source) hidden state
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    if options['use_dropout']:
        emb=emb*src_dropout
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder',
                                            mask=x_mask,
                                            emb_dropout=emb_dropout,
                                            rec_dropout=rec_dropout,
                                            )
    # word embedding for backward rnn (source) hidden state
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])
    if options['use_dropout']:
        embr = embr*src_dropout[::-1]
    projr = get_layer(options['encoder'])[1](tparams, embr, options,
                                             prefix='encoder_r',
                                             mask=xr_mask,
                                             emb_dropout=emb_dropout_r,
                                             rec_dropout=rec_dropout_r,
                                             )

    # context will be the concatenation of forward and backward rnns
    sourceMem = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

    # mean of the context (across time) will be used to initialize decoder rnn
    ctx_sum = (sourceMem * x_mask[:, :, None]).sum(0)
    if options['use_dropout']:
        ctx_sum *= shared_dropout_layer((n_samples,2*options['dim']), use_noise, trng, retain_hidden)
    
    ctx_mean = ctx_sum / x_mask.sum(0)[:, None]
    # or you can use the last state of forward + backward encoder rnns
    # ctx_mean = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)
    
    # initial decoder state
    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')
#     print 'init state dim ',init_state.ndim # dimension=2
    init_memory = get_layer('buffer_init')[1](tparams,ctx_sum,options,
                                    mask=x_mask,
                                    prefix='init_buffer',activ='tanh')

    # init_read = tensor.alloc(0., n_samples,options['buffer_size'])
    # word embedding (target), we will shift the target sequence one time step
    # to the right. This is done because of the bi-gram connections in the
    # readout and decoder rnn. The first target will be all zeros and we will
    # not condition on the last output.
    emb = tparams['Wemb_dec'][y.flatten()]
    emb = emb.reshape([n_timesteps_trg, n_samples, options['dim_word']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted
    if options['use_dropout']:
        emb *= trg_dropout

    # decoder - pass through the decoder conditional gru with attention
    proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                            prefix='decoder',
                                            mask=y_mask, context=sourceMem,
                                            context_mask=x_mask,
                                            one_step = False,
                                            init_memory = init_memory,
                                            init_state = init_state,
                                            emb_dropout=emb_dropout_d,
                                            ctx_dropout=ctx_dropout_d,
                                            rec_dropout=rec_dropout_d,
                                            buf_dropout=buf_dropout_d,
                                            )
    # (all time steps)hidden states of the decoder GRU: dimension=3
    proj_h = proj[0]
    # weighted averages of context, generated by attention module
    ctxs = proj[1]
    # weights (alignment matrix for cost) and weights(buffer alignment) 
    opt_ret['dec_alphas'] = proj[2]
    opt_ret['dec_alphas_buf'] = proj[4] # no use of this 

    # compute word probabilities
    logit_lstm = get_layer('ff')[1](tparams, proj_h, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)
    if options['use_dropout']:
        logit *= shared_dropout_layer((n_samples,options['dim_word']), use_noise, trng, retain_hidden)
    
    logit = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit', activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1],
                                               logit_shp[2]]))

    # cost
    y_flat = y.flatten()
    y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words'] + y_flat
    cost = -tensor.log(probs.flatten()[y_flat_idx])

    cost = cost.reshape([y.shape[0], y.shape[1]])
    cost = (cost * y_mask).sum(0)

    return trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost
# build force sampler
def build_force_sampler(tparams,options,use_noise):
    # build force-sample given inputs and outputs 
    # which is quite similar to that of build_model
    x = tensor.matrix('x', dtype='int64')
    y = tensor.matrix('y', dtype='int64')
    xr = x[::-1]

    n_timesteps = x.shape[0]
    n_timesteps_trg = y.shape[0]
    n_samples = x.shape[1]
    
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])
    if options['use_dropout']:
        # during test time create mean network as compensation 4 drop-out 
        retain_emb=1-options['dropout_emb']
        retain_src=1-options['dropout_src']
        retain_trg=1-options['dropout_trg']
        retain_hidden=1-options['dropout_hidden']
        # sentence to embedding
        src_dropout = theano.shared(numpy.float32(retain_src))
        trg_dropout = theano.shared(numpy.float32(retain_trg))
        emb *= src_dropout
        embr *= trg_dropout
        #in GRU,2 mask is needed(gate and state).BiRNN encoder is GRU
        rec_dropout = theano.shared(numpy.array([retain_hidden]*2,dtype='float32'))
        emb_dropout = theano.shared(numpy.array([retain_emb]*2,dtype='float32'))
        rec_dropout_r = theano.shared(numpy.array([retain_hidden]*2,dtype='float32'))
        emb_dropout_r = theano.shared(numpy.array([retain_emb]*2,dtype='float32'))
        #in decoder(cond_GRU),there are 2 GRU layers,and ctx is used in hidden state update and result 
        rec_dropout_d = theano.shared(numpy.array([retain_hidden]*10,dtype='float32'))
        emb_dropout_d = theano.shared(numpy.array([retain_emb]*2,dtype='float32'))
        ctx_dropout_d = theano.shared(numpy.array([retain_hidden]*4,dtype='float32'))
        buf_dropout_d = theano.shared(numpy.array([retain_hidden]*3,dtype='float32'))
    else:
        rec_dropout = theano.shared(numpy.array([1.]*2,dtype='float32'))
        emb_dropout = theano.shared(numpy.array([1.]*2,dtype='float32'))
        rec_dropout_r = theano.shared(numpy.array([1.]*2,dtype='float32'))
        emb_dropout_r = theano.shared(numpy.array([1.]*2,dtype='float32'))
        rec_dropout_d = theano.shared(numpy.array([1.]*10,dtype='float32'))
        emb_dropout_d = theano.shared(numpy.array([1.]*2,dtype='float32'))
        ctx_dropout_d = theano.shared(numpy.array([1.]*4,dtype='float32'))
        buf_dropout_d = theano.shared(numpy.array([1.]*3,dtype='float32'))
        
    # encoder
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder',
                                            emb_dropout=emb_dropout,
                                            rec_dropout=rec_dropout,)
    projr = get_layer(options['encoder'])[1](tparams, embr, options,
                                             prefix='encoder_r',
                                             emb_dropout=emb_dropout_r,
                                             rec_dropout=rec_dropout_r,)
    # concatenate forward and backward rnn hidden states
    sourceMem = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

    # get the input for decoder rnn initializer mlp
    ctx_sum = sourceMem.sum(0)
    if options['use_dropout']:
        ctx_sum*= retain_hidden
    ctx_mean = sourceMem.mean(0)
    
    # ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)
    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    init_memory = get_layer('buffer_init')[1](tparams,ctx_sum,options,
                                            timestep = sourceMem.shape[0],
                                            prefix='init_buffer',
                                            activ='tanh'
                                            )
    emb = tparams['Wemb_dec'][y.flatten()]
    emb = emb.reshape([n_timesteps_trg, n_samples, options['dim_word']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted
    if options['use_dropout']:
        emb *= trg_dropout

    # decoder - pass through the decoder conditional gru with attention
    proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                            prefix='decoder',
                                            mask=None, context=sourceMem,
                                            one_step = False,
                                            init_memory = init_memory,
                                            init_state = init_state,
                                            emb_dropout=emb_dropout_d,
                                            ctx_dropout=ctx_dropout_d,
                                            rec_dropout=rec_dropout_d,
                                            buf_dropout=buf_dropout_d,
                                            )
    # (all time steps)hidden states of the decoder GRU: dimension=3
    proj_h = proj[0]
    # weighted averages of context, generated by attention module
    ctxs = proj[1]
    # weights (alignment matrix for cost) and weights(buffer alignment) 
    attention = proj[2]
    buffer_weight = proj[4] 
    
    print 'Building force_decode_record...',
    inps = [x, y]
    outs = [buffer_weight,attention]
    force_decode_record=theano.function(inps, outs, name='force_decode_record', profile=profile)
    print 'Done'
    
    return force_decode_record

# build a sampler
def build_sampler(tparams, options, trng, use_noise):
    x = tensor.matrix('x', dtype='int64')
    xr = x[::-1]
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # word embedding (source), forward and backward
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])

    if options['use_dropout']:
        # during test time create mean network as compensation 4 drop-out 
        retain_emb=1-options['dropout_emb']
        retain_src=1-options['dropout_src']
        retain_trg=1-options['dropout_trg']
        retain_hidden=1-options['dropout_hidden']
        # sentence to embedding
        src_dropout = theano.shared(numpy.float32(retain_src))
        trg_dropout = theano.shared(numpy.float32(retain_trg))
        emb *= src_dropout
        embr *= trg_dropout
        #in GRU,2 mask is needed(gate and state).BiRNN encoder is GRU
        rec_dropout = theano.shared(numpy.array([retain_hidden]*2,dtype='float32'))
        emb_dropout = theano.shared(numpy.array([retain_emb]*2,dtype='float32'))
        rec_dropout_r = theano.shared(numpy.array([retain_hidden]*2,dtype='float32'))
        emb_dropout_r = theano.shared(numpy.array([retain_emb]*2,dtype='float32'))
        #in decoder(cond_GRU),there are 2 GRU layers,and ctx is used in hidden state update and result 
        rec_dropout_d = theano.shared(numpy.array([retain_hidden]*10,dtype='float32'))
        emb_dropout_d = theano.shared(numpy.array([retain_emb]*2,dtype='float32'))
        ctx_dropout_d = theano.shared(numpy.array([retain_hidden]*4,dtype='float32'))
        buf_dropout_d = theano.shared(numpy.array([retain_hidden]*3,dtype='float32'))
    else:
        rec_dropout = theano.shared(numpy.array([1.]*2,dtype='float32'))
        emb_dropout = theano.shared(numpy.array([1.]*2,dtype='float32'))
        rec_dropout_r = theano.shared(numpy.array([1.]*2,dtype='float32'))
        emb_dropout_r = theano.shared(numpy.array([1.]*2,dtype='float32'))
        rec_dropout_d = theano.shared(numpy.array([1.]*10,dtype='float32'))
        emb_dropout_d = theano.shared(numpy.array([1.]*2,dtype='float32'))
        ctx_dropout_d = theano.shared(numpy.array([1.]*4,dtype='float32'))
        buf_dropout_d = theano.shared(numpy.array([1.]*3,dtype='float32'))
        
    # encoder
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder',
                                            emb_dropout=emb_dropout,
                                            rec_dropout=rec_dropout,)
    projr = get_layer(options['encoder'])[1](tparams, embr, options,
                                             prefix='encoder_r',
                                             emb_dropout=emb_dropout_r,
                                             rec_dropout=rec_dropout_r,)

    # concatenate forward and backward rnn hidden states
    sourceMem = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

    # get the input for decoder rnn initializer mlp
    ctx_sum = sourceMem.sum(0)
    if options['use_dropout']:
        ctx_sum*= retain_hidden
    ctx_mean = sourceMem.mean(0)
    
    # ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)
    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    init_memory = get_layer('buffer_init')[1](tparams,ctx_sum,options,
                                            timestep = sourceMem.shape[0],
                                            prefix='init_buffer',
                                            activ='tanh'
                                            )
    # initiate elements for decoder: sample needs init_state init_memory and context(no need for init_read)
    print 'Building f_init...',
    outs = [init_state, sourceMem, init_memory]
    f_init = theano.function([x], outs, name='f_init', profile=profile)
    print 'Done'
#     theano.printing.pydotprint(init_memory, outfile='buffer.png',var_with_name_simple=False)
#     theano.printing.pydotprint(f_init, outfile="f_init.png", var_with_name_simple=False)

    # x: 1 x 1 declare input parameters for result generation
    y = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')
    init_memory = tensor.tensor3('init_memory', dtype='float32')
    init_read = tensor.matrix('init_read', dtype='float32')
    
    # if it's the first word, emb should be all zero and it is indicated by -1
    emb = tensor.switch(y[:, None] < 0,
                        tensor.alloc(0., 1, tparams['Wemb_dec'].shape[1]),
                        tparams['Wemb_dec'][y])
    if options['use_dropout']:
        emb*=trg_dropout
        
    # apply one step of conditional gru with attention
    proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                            prefix='decoder',
                                            mask=None, context=sourceMem,
                                            one_step=True,
                                            init_state=init_state,
                                            init_memory=init_memory,
                                            init_read= init_read,
                                            emb_dropout=emb_dropout_d,
                                            ctx_dropout=ctx_dropout_d,
                                            rec_dropout=rec_dropout_d,
                                            buf_dropout=buf_dropout_d,
                                            )
    # get the next hidden state
    next_state = proj[0]
    # get the weighted averages of context for this target word y
    # source reading results
    ctxs = proj[1]
    # get attention weight
    attention = proj[2]
    # get the next buffer 
    next_buffer = proj[3]
    # get the next buffer reading weight 
    next_read = proj[4]

    if options['use_dropout']:
        next_state_new=next_state * retain_hidden
        emb*=retain_emb
        ctxs*=retain_hidden
    else:
        next_state_new=next_state

    logit_lstm = get_layer('ff')[1](tparams, next_state_new, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)
    if options['use_dropout']:
        logit *=retain_hidden
    logit = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit', activ='linear')

    # compute the softmax probability
    next_probs = tensor.nnet.softmax(logit)

    # sample from softmax distribution to get the sample
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # compile a function to do the whole thing above, next word probability,
    # sampled word for the next target, next hidden state, next buffer_alpha
    # to be used
    print 'Building f_next..',
    inps = [y, sourceMem, init_state, init_memory, init_read]
    outs = [next_probs, next_sample, next_state, next_buffer, next_read, attention]
    f_next = theano.function(inps, outs, name='f_next', profile=profile)
    print 'Done'

    return f_init, f_next


# generate sample, either with stochastic sampling or beam search. Note that,
# this function iteratively calls f_init and f_next functions.
def gen_sample(tparams, f_init, f_next, x, options, trng=None, k=1, maxlen=30,
               stochastic=True, argmax=False, return_attention=False, 
               analyseBuffer=False, normalize=True):

    # k is the beam size we have
    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    alpha_buf_record = []
    attention_record = []
    # if training, return best sample and score
    if stochastic:
        sample_score = 0

    # for beam search states
    live_k = 1 
    dead_k = 0 # result ends , consume beam slot
    hyp_samples = [[]] * live_k
    if analyseBuffer:
        hyp_alpha_buf_record = [[]] * live_k
    if return_attention:
        hyp_attention_record = [[]] * live_k
    
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = [] 
    hyp_buffers = []
    hyp_reads = []

    # get initial state of decoder rnn and encoder context
    # initial state, source memory and initial buffer
    ret = f_init(x)
    next_w = -1 * numpy.ones((1,)).astype('int64')  # bos indicator
    next_state, ctx0, next_buffer = ret[0], ret[1], ret[2]
    next_read = numpy.zeros((1,options['buffer_size'])).astype('float32')
    
#     print 'genSample: '
    for ii in xrange(maxlen): # for max sentence length 
        ctx = numpy.tile(ctx0, [live_k, 1])
        inps = [next_w, ctx, next_state, next_buffer, next_read]
        ret = f_next(*inps)
        next_p, next_w, next_state ,next_buffer, next_read, attention=\
         ret[0], ret[1], ret[2],ret[3], ret[4], ret[5]
        
        if stochastic:
            # in train: get the most possible word as 'nw' appended in sample result
            if argmax:
                nw = next_p[0].argmax()
            else:
                nw = next_w[0]
            sample.append(nw)
            sample_score -= numpy.log(next_p[0, nw])
            if nw == 0: # end of sentence
                break
        else:
            # in translate: k beam search sort top k word index and costs
            cand_scores = hyp_scores[:, None] - numpy.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)] # top k samples
            
            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            if analyseBuffer:
                new_alpha_buf_record = []
            if return_attention:    
                new_attention_record = []
            
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_hyp_states = []
            new_hyp_buffers = []
            new_hyp_reads = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                if analyseBuffer:    
                    new_alpha_buf_record.append(hyp_alpha_buf_record[ti]+[next_read[ti]])
                if return_attention:    
                    new_attention_record.append(hyp_attention_record[ti]+[attention[ti]])
                # extract top k candidates(cost and p and sample word)
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(copy.copy(next_state[ti]))
                new_hyp_buffers.append(copy.copy(next_buffer[:,ti,:]))
                new_hyp_reads.append(copy.copy(next_read[ti]))

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            if analyseBuffer:
                hyp_alpha_buf_record = []
            if return_attention:
                hyp_attention_record = []
            hyp_scores = []
            hyp_states = []
            hyp_buffers = []
            hyp_reads = []

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                # top choice is end of sentence, beam size decreased
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    if analyseBuffer:    
                        alpha_buf_record.append(new_alpha_buf_record[idx])
                    if return_attention:    
                        attention_record.append(new_attention_record[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    if analyseBuffer:    
                        hyp_alpha_buf_record.append(new_alpha_buf_record[idx])
                    if return_attention:    
                        hyp_attention_record.append(new_attention_record[idx])
                    
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
                    hyp_buffers.append(new_hyp_buffers[idx][:,None,:])
                    hyp_reads.append(new_hyp_reads[idx])
                    
            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break
            
            # concatenate top k candidates on 'n_samples' axis
            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = numpy.array(hyp_states)
            next_buffer = numpy.concatenate(hyp_buffers, axis=1)
            next_read = numpy.array(hyp_reads)

    if not stochastic:
        # in translate: dump every remaining one and return the samples
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])
                # record buffer reading results:
                if analyseBuffer:
                    alpha_buf_record.append(hyp_alpha_buf_record[idx])
                if return_attention:
                    attention_record.append(hyp_attention_record[idx])
    
    if not analyseBuffer:
        alpha_buf_record=None
    if not return_attention:
        attention_record=None
        
    return sample, sample_score, alpha_buf_record, attention_record


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
    b2 = 0.001
    e = 1e-8

    updates = []

    i = theano.shared(numpy.float32(0.))
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr0 * (tensor.sqrt(fix2) / fix1)

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

def train(dim_word=100,  # word vector dimensionality
          dim=1000,  # the number of LSTM units
          encoder='gru',
          decoder='memDec',
          patience=10,  # early stopping patience
          patience_bleu=50, # early stopping patience for bleu
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100, # update info display frequency
          decay_c=0.,  # L2 regularization penalty
          alpha_c=0.,  # alignment regularization
          clip_c=-1.,  # gradient clipping threshold
          lrate=0.01,  # learning rate
          n_words_src=100000,  # source vocabulary size
          n_words=100000,  # target vocabulary size
          buffer_size=8,
          buffer_dim=1000,
          maxlen=100,  # maximum length of the description
          optimizer='rmsprop',
          batch_size=16,
          valid_batch_size=16,
          saveto='model.npz',
          validFreq=1000,
          saveFreq=1000,   # save the parameters after every saveFreq updates
          sampleFreq=100,   # generate some samples after every sampleFreq
          datasets=[
              '/data/cn.tok',
              '/data/en.tok'],
          valid_datasets=['../data/MT06.cn.txt',
                          '../data/MT06.en.txt'],
          dictionaries=[
              '/data/cn.tok.pkl',
              '/data/en.tok.pkl'],
          use_dropout=False,
          dropout_emb=0.1, # dropout rate on layers(used in layer init)
          dropout_hidden=0.2, 
          dropout_src=0.0,
          dropout_trg=0.0,
          reload_=False,
          overwrite=False,
          **bleu_params
          ):

    # Model options
    model_options = locals().copy()
    # BLEU validation #
    bleu_valid = BleuValidator(model_options, **bleu_params)

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
    else:
        print 'Mind: memDec is better with attention model initiation'

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
#     theano.printing.pydotprint(f_cost, outfile="f_cost.png", var_with_name_simple=False)
#     theano.printing.debugprint(cost,)
#     print itemlist(tparams)
    
    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
#     theano.printing.pydotprint(grads,outfile="grad.png",var_with_name_simple=False)
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
    bad_counter_bleu = 0
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
            if numpy.mod(uidx, dispFreq) == 0: # 
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud
            # save the best model so far, in addition, save the latest model
            # into a separate file with the iteration number for external eval
##############################################################################################
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
##############################################################################################
            # generate some samples with the model and display them
            if numpy.mod(uidx, sampleFreq) == 0:
                # FIXME: random selection
                for jj in xrange(numpy.minimum(5, x.shape[1])):
                    stochastic = True
                    sampleData = gen_sample(tparams, f_init, f_next,
                                               x[:, jj][:, None],
                                               model_options, trng=trng, k=1,
                                               maxlen=30,
                                               stochastic=stochastic,
                                               argmax=False)
                    sample=sampleData[0]
                    score=sampleData[1]
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
##############################################################################################
            # validate model on validation set and early stop if necessary
            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                valid_errs = pred_probs(f_log_probs, prepare_data,
                                        model_options, valid)
                valid_err = valid_errs.mean()
                history_errs.append(valid_err)

            # bleu validation only happends when valid cost came below 100
            # or the bleu is always 0 then early stops when training starts
                if numpy.mod(uidx,validFreq*10)==0 and valid_err<70:
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
                            print 'remove temp file... '+previous_model,
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
                        best_p = unzip(tparams)
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
##############################################################################################
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
                uidx=uidx,
                **params)

    return valid_err

if __name__ == '__main__':
    pass
