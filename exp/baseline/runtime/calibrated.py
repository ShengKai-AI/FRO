"""Label-blind per-input FLOPs matching against FRO's fixed retained budget."""
import time
import torch
from contextlib import contextmanager
from exp.baseline.metrics.operations import MatrixCounter
from exp.baseline.metrics.flops import decoder_flops,fro_extra
from exp.baseline.methods.random import compress as random_compress
from exp.baseline.methods.video import apply_video

@contextmanager
def reuse_encoders(thinker,stream):
    # A calibration retry changes selection only; retain the exact encoder outputs.
    saved=[];cache=stream.setdefault('_encoder_outputs',{})
    for name in ('visual','audio_tower'):
        module=getattr(thinker,name,None)
        if module is None:continue
        original=module.forward;saved.append((module,original))
        def forward(*args,_name=name,_original=original,**kwargs):
            if _name not in cache:
                cache[_name]=_original(*args,**kwargs)
                owner=getattr(thinker,_name)
                cache[_name+'_stats']={k:getattr(owner,k) for k in ('_vz_attn_mean','_vz_attn_key','_fro_encoder_aux') if hasattr(owner,k)}
                if _name=='audio_tower' and hasattr(thinker,'_omnizip_audio_attn'):cache['omnizip_attention']=thinker._omnizip_audio_attn
            else:
                for k,v in cache.get(_name+'_stats',{}).items():setattr(getattr(thinker,_name),k,v)
                if _name=='audio_tower' and 'omnizip_attention' in cache:thinker._omnizip_audio_attn=cache['omnizip_attention']
            return cache[_name]
        module.forward=forward
    try:yield
    finally:
        for module,forward in saved:module.forward=forward

def dimensions(ns,data,anchor):
    thinker=ns['thinker'];c=thinker.config.text_config.to_dict();counts=[];lengths=[];queries=[];anchors=[]
    for s in data['streams']:
        ids=s['inputs']['input_ids'][0];av=(ids==thinker.config.audio_token_id)|(ids==thinker.config.video_token_id)
        mods=[int((ids==getattr(thinker.config,k+'_token_id')).sum()) for k in ('audio','video')]
        counts.extend(mods);lengths.append(len(ids));anchors.append(sum(min(16,n) for n in mods))
        q=((torch.arange(len(ids))>int(av.nonzero()[-1]))&(ids<151643)).sum()
        queries.append(min(32,int(q)))
    ids=data['joint']['input_ids'][0];av=sum(counts);text=len(ids)-av
    extras=fro_extra(c,lengths,[sum(counts[:2]),sum(counts[2:])],queries,anchors)
    retained=sum(max(1,round(n*anchor)) if n else 0 for n in counts)
    target=decoder_flops(c,[text+retained]*c['num_hidden_layers'],text+retained,extras=extras,auxiliary_delta=0)['total']
    return c,counts,text,extras,target

def estimate(c,method,ratio,counts,text,aux=0):
    kept=sum(max(1,round(n*ratio)) if n else 0 for n in counts);small=text+kept;L=c['num_hidden_layers'];lens=[small]*L
    if method=='fastv':lens=[text+sum(counts)]*5+[small]*(L-5)
    elif method=='omnipack':
        pre=text+sum(max(1,round(n*min(1,ratio*2))) if n else 0 for n in counts);cut=26 if L!=28 else 18
        lens=[pre]*cut+[small]*(L-cut)
    return decoder_flops(c,lens,small,kv_lengths=lens,auxiliary_delta=aux)['total']

def solve(c,method,target,counts,text,aux=0):
    lo,hi=.00001,.5 if method=='omnipack' else 1.
    if estimate(c,method,lo,counts,text,aux)>target:
        raise ValueError('Minimum attainable method FLOPs exceed the FRO target')
    for _ in range(24):
        mid=(lo+hi)/2
        if estimate(c,method,mid,counts,text,aux)<=target:lo=mid
        else:hi=mid
    return lo

def select(ns,data,method,ratio,seed):
    from exp.baseline.worker import compress_native
    from fro.scoring import ScoreConfig,prune
    from fro.temporal import attach_metadata
    thinker=ns['thinker'];parts=[];aux=0
    if method=='omnipack':
        thinker.omnipack_config.video_encoder_ratio=thinker.omnipack_config.audio_encoder_ratio=min(1.,2*ratio)
    for i,s in enumerate(data['streams']):
        counter=MatrixCounter()
        encoders=[getattr(thinker,k) for k in ('visual','audio_tower') if hasattr(thinker,k)]
        with reuse_encoders(thinker,s),counter,counter.exclude(encoders):
            part=compress_native(ns,s,method,ratio,counter=counter)
        # Native retention search records the chosen invocation, not discarded trials.
        aux+=part.get('aux',{}).get('selected_auxiliary_flops',counter.total)
        aux+=sum(getattr(e,'_fro_encoder_aux',0) for e in encoders)
        if method in ('visionzip','fastvid'):
            part['vision_attention']=thinker.visual._vz_attn_mean.clone();part['vision_keys']=thinker.visual._vz_attn_key.clone()
        if method=='random':part=random_compress(part,thinker,ratio,seed+i)
        elif method in ('visionzip','fastvid','vidcom2','dycoke_ttm'):
            part=apply_video(part,s,thinker,method,ratio,seed+i);aux+=part['aux']['selected_auxiliary_flops']
        parts.append(part)
    if method=='fro':
        cfg=ScoreConfig(module='temporal_novelty',bridge_mode='same');attach_metadata(parts,data,ns['p'].tokenizer,cfg.module)
        parts=prune(parts,thinker,cfg,ratio)
    return parts,aux

@torch.inference_mode()
def run_pair(ns,data,method,anchor,seed,model,tolerance=.005):
    thinker=ns['thinker'];c,counts,text,extras,target=dimensions(ns,data,anchor)
    ids=data['joint']['input_ids'][0];av=(ids==thinker.config.audio_token_id)|(ids==thinker.config.video_token_id)
    ratio=anchor if method=='fro' else solve(c,method,target,counts,text)
    trials=[];started=time.perf_counter();best=None;extra=extras if method=='fro' else {}
    for attempt in range(10):
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        parts,aux=select(ns,data,method,ratio,seed)
        emb,pos,keep=ns['joint_features'](data,parts);stage1=keep.clone()
        if method in ('omnipack','fastv'):
            selected_ids=ids[keep].cuda();am=selected_ids==thinker.config.audio_token_id;vm=selected_ids==thinker.config.video_token_id
            if method=='omnipack':
                thinker.model.omnipack_audio_token_mask=am;thinker.model.omnipack_video_token_mask=vm;thinker.model.omnipack_text_token_mask=~(am|vm)
                thinker.model.omnipack_global_config={'stage2_layer':26 if model=='3B' else 18,'video_stage2_keep_ratio':.5,'audio_stage2_keep_ratio':.5}
            else:
                thinker.model.fastv_global_audio_token_mask=am;thinker.model.fastv_global_video_token_mask=vm;thinker.model.fastv_global_text_token_mask=~(am|vm)
                thinker.model.fastv_global_config={'method_name':'fastv_omni','video_ratio':ratio,'audio_ratio':ratio,'fastv_k':5}
        # Pre-LLM methods can calibrate without replaying the decoder.
        if method not in ('omnipack','fastv'):
            predicted=decoder_flops(c,[int(keep.sum())]*c['num_hidden_layers'],int(keep.sum()),extras=extra,auxiliary_delta=aux)
            error=predicted['total']/target-1
            if abs(error)>tolerance and attempt<9:
                trials.append({'ratio':ratio,'flops':predicted['total'],'relative_error':error,'decoded':False,'auxiliary_flops':aux})
                estimate_new=solve(c,method,target,counts,text,aux) if method!='fro' else anchor
                # Correct attainable native count rounding based on the measured cost slope.
                slope=(estimate(c,method,min(1,ratio+.001),counts,text,aux)-estimate(c,method,max(.00001,ratio-.001),counts,text,aux))/.002
                ratio=max(.00001,min(.999,ratio-error*target/max(1,slope)))
                del parts,emb,pos
                continue
        ns['state']['inner_auxiliary_flops']=0
        torch.cuda.synchronize();selected=time.perf_counter()
        answer,tokens,first,lens,stop=ns['decode'](emb,pos)
        torch.cuda.synchronize()
        if method in ('omnipack','fastv'):
            keep=stage1.clone();keep[stage1]=ns['state']['joint_stage2'].cpu()
        aux+=ns['state'].get('inner_auxiliary_flops',0)
        flops=decoder_flops(c,lens,int(keep.sum()),kv_lengths=ns['state']['layer_kv_prefill_lengths'],extras=extra,auxiliary_delta=aux)
        error=flops['total']/target-1
        trials.append({'ratio':ratio,'flops':flops['total'],'relative_error':error,'decoded':True,'auxiliary_flops':aux})
        if not bool(keep[~av].all()):raise ValueError('Text token removed')
        best=dict(status='passed' if abs(error)<=tolerance else 'budget_unmatched',answer=answer,generated_tokens=len(tokens),seed=seed,
            target_ratio=ratio,fro_anchor_retention=anchor,flops_target=target,flops_relative_error=error,calibration_trials=trials.copy(),
            final_av_ratio=float(keep[av].sum()/av.sum()),original_tokens=len(ids),pre_llm_tokens=int(stage1.sum()),final_tokens=int(keep.sum()),
            layer_prefill_lengths=lens,pre_llm_seconds=selected-started,ttft_seconds=first-started,elapsed_seconds=time.perf_counter()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,flops=flops,
            stream_diagnostics=[p.get('score_diagnostics',p.get('aux',{})) for p in parts],
            modality_counts={k:{'before':int((ids==getattr(thinker.config,k+'_token_id')).sum()),'final':int(keep[ids==getattr(thinker.config,k+'_token_id')].sum())} for k in ('audio','video')})
        del parts,emb,pos
        if best['status']=='passed':break
        slope=(estimate(c,method,min(.5 if method=='omnipack' else 1.,ratio+.001),counts,text,aux)-estimate(c,method,max(.00001,ratio-.001),counts,text,aux))/.002
        ratio=max(.00001,min(.5 if method=='omnipack' else .999,ratio-error*target/max(1,slope)))
    ns['state'].clear()
    if best is None:raise RuntimeError('No feasible calibrated decoder invocation')
    return best
