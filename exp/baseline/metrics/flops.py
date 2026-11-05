"""v5 dense-matrix FLOPs convention, 1 MAC = 2 FLOPs.

Matches the reviewed plan exactly at its reference input. Real runs pass measured
per-layer prefill and KV lengths. LM head, encoders and elementwise operators are
excluded from FLOPs-dagger; unmeasured auxiliary delta is never silently zeroed.
"""
from contextvars import ContextVar
DECODE_STEPS=ContextVar("decode_steps",default=100)

def coefficient(c):
    d=c['hidden_size'];k=d//c['num_attention_heads']*c['num_key_value_heads']
    return 4*d*d+4*d*k+6*d*c['intermediate_size']

def prefill(c,n):return coefficient(c)*n+4*c['hidden_size']*n*n

def decode(c,n,steps=100):
    return steps*coefficient(c)+4*c['hidden_size']*(steps*n+steps*(steps+1)//2)

def fro_extra(c,stream_lengths,av_counts,query_counts=None,anchor_counts=None):
    d=c['hidden_size'];k=d//c['num_attention_heads']*c['num_key_value_heads']
    queries=query_counts or [32]*len(stream_lengths);anchors=anchor_counts or [32]*len(stream_lengths)
    return {'fro_prefix':sum(prefill(c,n) for n in stream_lengths),
            'fro_qk':sum(2*q*d*d+2*n*d*k+2*q*n*d for n,q in zip(stream_lengths,queries)),
            'fro_cross':sum(2*av*anchors[1-i]*d for i,av in enumerate(av_counts))}

def decoder_flops(config,layer_lengths,final_tokens,decode_tokens=None,kv_lengths=None,extras=None,auxiliary_delta=None):
    decode_tokens=DECODE_STEPS.get() if decode_tokens is None else decode_tokens
    if len(layer_lengths)!=config['num_hidden_layers']:raise ValueError('Layer count mismatch')
    kv_lengths=kv_lengths if kv_lengths is not None else [final_tokens]*len(layer_lengths)
    if len(kv_lengths)!=len(layer_lengths):raise ValueError('Missing KV instrumentation')
    p=sum(prefill(config,n) for n in layer_lengths);d=sum(decode(config,n,decode_tokens) for n in kv_lengths)
    extras=extras or {};dagger=p+d+sum(extras.values())
    return {'convention':f'dense_matrix_{decode_tokens}_decode_steps','decoder_prefill':p,f'decoder_decode_{decode_tokens}':d,'decoder_total':p+d,
            **extras,'flops_dagger':dagger,'auxiliary_delta':auxiliary_delta,'total':None if auxiliary_delta is None else dagger+auxiliary_delta,
            'total_complete':auxiliary_delta is not None,'layer_kv_lengths':kv_lengths,'excluded':['encoders','lm_head','softmax','normalization','sorting','elementwise operators']}

def reference_cost(c,method,rate,av=32000,text=256):
    L=c['num_hidden_layers'];n=av+text;small=text+round(av*rate);lengths=[small]*L
    if method=='full_tokens':lengths=[n]*L
    elif method=='fastv':lengths=[n]*5+[small]*(L-5)
    elif method=='omnipack':
        cut=18 if L==28 else 26;lengths=[text+round(av*rate*2)]*cut+[small]*(L-cut)
    extras=fro_extra(c,[n//2,n//2],[av//2,av//2]) if method=='fro' else {}
    return decoder_flops(c,lengths,small,kv_lengths=lengths,extras=extras)['flops_dagger']

def solve_reference_retention(c,method,anchor_rate,av=32000,text=256):
    target=reference_cost(c,'fro',anchor_rate,av,text)
    if method=='fro':return anchor_rate,target
    if method=='full_tokens':return 1.,reference_cost(c,method,1.,av,text)
    lo,hi=0,av//2 if method=='omnipack' else av
    while lo<hi:
        mid=(lo+hi+1)//2
        if reference_cost(c,method,mid/av,av,text)<=target:lo=mid
        else:hi=mid-1
    return lo/av,target
