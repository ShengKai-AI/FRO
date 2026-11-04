"""Analytical decoder FLOPs; 1 multiply-add = 2 FLOPs.

Dense Q/K/V/O + SwiGLU and causal attention. Norm, RoPE, softmax and
selection/auxiliary models are excluded from this component, never counted as zero
in the total. Decode uses final retained KV lengths (all-layer cache pruning).
"""
def layer_flops(config, n, context=None):
    d=config['hidden_size']; h=config['num_attention_heads']; kv=config['num_key_value_heads']
    hd=config.get('head_dim',d//h); inter=config['intermediate_size']
    projections=2*n*(d*h*hd+2*d*kv*hd+h*hd*d+3*d*inter)
    pairs=n*(n+1)//2 if context is None else n*context
    return projections+4*pairs*h*hd

def decoder_flops(config, layer_lengths, final_tokens, decode_tokens=100):
    if len(layer_lengths)!=config['num_hidden_layers']:raise ValueError('Layer count mismatch')
    # One LM-head projection on prefill; subsequent decode steps produce remaining tokens.
    prefill=sum(layer_flops(config,n) for n in layer_lengths)
    decode=sum(config['num_hidden_layers']*layer_flops(config,1,final_tokens+i) for i in range(1,decode_tokens))
    head=2*decode_tokens*config['hidden_size']*config['vocab_size']
    return {'decoder_prefill':prefill,'decoder_decode_100':decode,'lm_head':head,'decoder_total':prefill+decode+head,
            'selection_and_auxiliary':None,'total':None,'total_complete':False,
            'accounting':'analytical; excludes norm/softmax/RoPE; auxiliary/selection FLOPs unmeasured'}
