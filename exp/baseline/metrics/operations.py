"""Count executed auxiliary dense products (MAC=2), without common encoders.

Decoder attention FLOPs are analytical; this context covers selection/AudioCLIP.
Elementwise operations, normalization, FFT and sorting follow the plan exclusions.
"""
import math
from collections import Counter
from contextlib import contextmanager
from torch.utils._python_dispatch import TorchDispatchMode

class MatrixCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__();self.counts=Counter();self.suspended=0
    @property
    def total(self):return sum(self.counts.values())
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        out=func(*args,**(kwargs or {}))
        if self.suspended:return out
        name=func._schema.name;n=0
        if name in ('aten::mm','aten::bmm','aten::matmul','aten::linear'):
            n=2*out.numel()*args[0].shape[-1]
        elif name in ('aten::cdist','aten::_cdist_forward'):
            n=2*out.numel()*args[0].shape[-1]
        elif name in ('aten::addmm','aten::baddbmm'):
            n=2*out.numel()*args[1].shape[-1]
        elif name in ('aten::conv1d','aten::conv2d','aten::conv3d'):
            n=2*out.numel()*math.prod(args[1].shape[1:])
        elif name in ('aten::convolution','aten::_convolution'):
            if args[6]:raise ValueError('Transposed convolution requires explicit FLOPs accounting')
            n=2*out.numel()*math.prod(args[1].shape[1:])
        elif name in ('aten::_scaled_dot_product_flash_attention','aten::_scaled_dot_product_efficient_attention','aten::_scaled_dot_product_flash_attention_for_cpu','aten::_scaled_dot_product_cudnn_attention','aten::scaled_dot_product_attention'):
            q,k,v=args[:3];n=2*math.prod(q.shape[:-2])*q.shape[-2]*k.shape[-2]*(q.shape[-1]+v.shape[-1])
        elif name=='aten::_native_multi_head_attention':
            raise ValueError('Fused MHA must be disabled for auditable auxiliary FLOPs')
        if n:self.counts[name]+=int(n)
        return out
    @contextmanager
    def exclude(self,modules):
        handles=[]
        def before(*_):self.suspended+=1
        def after(*_):self.suspended-=1
        for module in modules:
            handles.append(module.register_forward_pre_hook(before))
            handles.append(module.register_forward_hook(after,always_call=True))
        try:yield
        finally:
            for h in handles:h.remove()
