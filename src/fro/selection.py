"""Source-aware joint token selection. No dataset labels or evidence annotations.
Question attention is computed in the native next-block Q/K space with native mRoPE.
Cross-source support is a hypothesis, measured by source-aware ablations, not assumed
semantic alignment. All budgets count actual AV tokens; all text tokens survive.
"""
from dataclasses import dataclass,asdict
import math
import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class CompressionConfig:
    method: str = 'joint'
    ratio: float = .5
    layer: int = 2
    bridge_weight: float = .3
    anchors_per_group: int = 16
    coverage_fraction: float = .2
    adaptive_budget: bool = False
    seed: int = 20261030


def rank01(x):
    order=torch.argsort(x,stable=True);r=torch.empty_like(x,dtype=torch.float32)
    r[order]=torch.arange(len(x),device=x.device,dtype=torch.float32)
    return r/max(1,len(x)-1)


def native_question_attention(layer,h,pos_embeddings,query_positions,rotate,repeat_kv):
    """Mean true causal attention over sampled question tokens/heads, without N*N map."""
    attn=layer.self_attn;n=h.shape[1]
    qpos=query_positions
    if len(qpos)>32:qpos=qpos[torch.linspace(0,len(qpos)-1,32,device=h.device).round().long()]
    norm=layer.input_layernorm(h)
    q=attn.q_proj(norm[:,qpos]).view(1,len(qpos),attn.num_heads,attn.head_dim).transpose(1,2)
    k=attn.k_proj(norm).view(1,n,attn.num_key_value_heads,attn.head_dim).transpose(1,2)
    cos,sin=pos_embeddings
    q,_=rotate(q,q,cos[:,:,qpos],sin[:,:,qpos],attn.rope_scaling['mrope_section'])
    k,_=rotate(k,k,cos,sin,attn.rope_scaling['mrope_section'])
    k=repeat_kv(k,attn.num_key_value_groups)
    # Bound temporary storage; no full-sequence square attention matrix.
    score=torch.zeros(n,device=h.device,dtype=torch.float32)
    future=torch.arange(n,device=h.device)[None,:]>qpos[:,None]
    for start in range(0,attn.num_heads,4):
        logits=(q[:,start:start+4].float()@k[:,start:start+4].float().transpose(-1,-2))/math.sqrt(attn.head_dim)
        logits.masked_fill_(future[None,None],float('-inf'))
        score+=torch.softmax(logits,-1).sum((0,1,2))
    score/=attn.num_heads*len(qpos)
    return score,qpos


def allocate_budget(counts,total,mass=None):
    """Exact integer budget with per-source/modality floors and capacity constraints."""
    ns=counts.to(torch.long);N=int(ns.sum());total=max(4,min(int(total),N))
    if total==N:return ns.clone()
    floors=torch.minimum(ns,torch.maximum(torch.ones_like(ns),torch.floor(ns.float()*.1).long()))
    if int(floors.sum())>total:floors=torch.minimum(ns,torch.ones_like(ns))
    alloc=floors.clone();remaining=total-int(alloc.sum())
    base=ns.float()/ns.sum()
    weights=base if mass is None else .5*base+.5*mass.float().clamp_min(1e-12)/mass.sum().clamp_min(1e-12)
    for _ in range(8):
        if remaining<=0:break
        room=ns-alloc;active=room>0;w=weights*active;w=w/w.sum()
        raw=w*remaining;add=torch.minimum(room,raw.floor().long());alloc+=add;remaining=total-int(alloc.sum())
        if remaining<=0:break
        # At most four rounding remainders per iteration; capacities may require another round.
        priority=(raw-raw.floor()).masked_fill((ns-alloc)<=0,-1)
        for idx in torch.argsort(priority,descending=True,stable=True).tolist():
            if remaining<=0:break
            if int(alloc[idx])<int(ns[idx]):alloc[idx]+=1;remaining-=1
    assert int(alloc.sum())==total and bool(torch.all(alloc<=ns))
    return alloc


def select_with_coverage(score,k,coverage_fraction):
    n=len(score)
    if k>=n:return torch.ones(n,device=score.device,dtype=torch.bool)
    keep=torch.zeros(n,device=score.device,dtype=torch.bool)
    bins=max(1,min(n,k,int(round(k*coverage_fraction))))
    group=torch.arange(n,device=score.device)*bins//n
    best=torch.full((bins,),-torch.inf,device=score.device)
    best.scatter_reduce_(0,group,score.float(),reduce='amax',include_self=True)
    candidate=torch.where(score==best[group],torch.arange(n,device=score.device),n)
    idx=torch.full((bins,),n,device=score.device,dtype=torch.long)
    idx.scatter_reduce_(0,group,candidate,reduce='amin',include_self=True)
    keep[idx]=True
    need=k-int(keep.sum())
    if need:keep[torch.topk(score.masked_fill(keep,-torch.inf),need,sorted=False).indices]=True
    assert int(keep.sum())==k
    return keep


@torch.inference_mode()
def select_tokens(h,groups,question_attention,cfg,cross=None):
    n=len(groups);av=groups>=0;keep=~av
    counts=torch.stack([(groups==g).sum() for g in range(4)])
    assert bool(torch.all(counts>0))
    target=round(float(cfg.ratio)*int(counts.sum()))
    raw=question_attention.float()
    mass=torch.stack([raw[groups==g].sum() for g in range(4)])
    budgets=allocate_budget(counts,target,mass if cfg.adaptive_budget else None)
    relevance=torch.zeros(n,device=h.device);bridge=torch.zeros_like(relevance);partner=torch.full((n,),-1,device=h.device,dtype=torch.long)
    affinity=torch.zeros_like(relevance);anchor_sets=[]
    for g in range(4):
        ids=(groups==g).nonzero().flatten();relevance[ids]=rank01(raw[ids])
        anchors=ids[torch.topk(raw[ids],min(cfg.anchors_per_group,len(ids))).indices];anchor_sets.append(anchors)
    if cfg.method in ['joint','joint_shuffle']:
        # Source-aware, modality-centered feature affinity. Centering reduces the
        # common modality direction; links do not imply synchronized timestamps.
        z=h[0].float()
        z=z-z.mean(-1,keepdim=True)
        for mod in [0,1]:
            ids=((groups==mod)|(groups==mod+2)).nonzero().flatten()
            z[ids]=z[ids]-z[ids].mean(0,keepdim=True)
        z=F.normalize(z,dim=-1).to(h.dtype)
        for g in range(4):
            ids=(groups==g).nonzero().flatten();other=[2,3] if g<2 else [0,1]
            anchors=torch.cat([anchor_sets[a] for a in other])
            sim=(z[ids]@z[anchors].T).float()
            # Query-relevant anchors can support complementary evidence in the other stream.
            weights=.5+.5*relevance[anchors]
            support=(sim.clamp_min(0)*weights[None,:])
            val,which=support.max(-1);partner[ids]=anchors[which];affinity[ids]=sim.gather(1,which[:,None]).flatten()
            b=rank01(val)
            if cfg.method=='joint_shuffle':
                gen=torch.Generator(device=h.device).manual_seed(cfg.seed+g)
                b=b[torch.randperm(len(ids),generator=gen,device=h.device)]
            bridge[ids]=b
    final_score=relevance.clone()
    if cfg.method in ['joint','joint_shuffle']:
        final_score=(1-cfg.bridge_weight)*relevance+cfg.bridge_weight*bridge
    if cfg.method in ['joint_attention','shuffled_attention']:
        assert cross is not None
        partner=cross['partner'];affinity=cross['affinity']
        for g in [0,1]:
            ids=(groups==g).nonzero().flatten();b=rank01(cross['cross_score'][ids])
            if cfg.method=='shuffled_attention':
                gen=torch.Generator(device=h.device).manual_seed(cfg.seed+g)
                b=b[torch.randperm(len(ids),generator=gen,device=h.device)]
            bridge[ids]=b;final_score[ids]=(1-cfg.bridge_weight)*relevance[ids]+cfg.bridge_weight*b
    for g in range(4):
        ids=(groups==g).nonzero().flatten();k=int(budgets[g])
        if cfg.method=='uniform':
            local=torch.zeros(len(ids),device=h.device,dtype=torch.bool);local[torch.linspace(0,len(ids)-1,k,device=h.device).round().long()]=True
        elif cfg.method=='random':
            gen=torch.Generator(device=h.device).manual_seed(cfg.seed+g)
            local=torch.zeros(len(ids),device=h.device,dtype=torch.bool);local[torch.randperm(len(ids),generator=gen,device=h.device)[:k]]=True
        else:local=select_with_coverage(final_score[ids],k,cfg.coverage_fraction)
        keep[ids]=local
    assert int((keep&av).sum())==target and bool(keep[~av].all())
    return keep,{'attention':raw,'relevance':relevance,'bridge':bridge,'score':final_score,'partner':partner,'affinity':affinity,'budgets':budgets,'counts':counts}


def native_cross_support(layer,h,pos_embeddings,groups,question_attention,anchors_per_group,rotate,repeat_kv):
    """Causal two-hop attention proxy: question -> later source -> earlier source.
    It respects actual transformer causality; no invented simultaneous alignment or
    reverse attention is attributed to the model. Returns link provenance.
    """
    a=layer.self_attn;n=h.shape[1];anchors=[]
    for g in [2,3]:
        ids=(groups==g).nonzero().flatten();anchors.append(ids[torch.topk(question_attention[ids],min(anchors_per_group,len(ids))).indices])
    qpos=torch.cat(anchors);target=(groups//2==0).nonzero().flatten()
    norm=layer.input_layernorm(h)
    q=a.q_proj(norm[:,qpos]).view(1,len(qpos),a.num_heads,a.head_dim).transpose(1,2)
    k=a.k_proj(norm).view(1,n,a.num_key_value_heads,a.head_dim).transpose(1,2)
    cos,sin=pos_embeddings
    q,_=rotate(q,q,cos[:,:,qpos],sin[:,:,qpos],a.rope_scaling['mrope_section']);k,_=rotate(k,k,cos,sin,a.rope_scaling['mrope_section']);k=repeat_kv(k,a.num_key_value_groups)
    links=torch.zeros(len(qpos),len(target),device=h.device,dtype=torch.float32)
    future=torch.arange(n,device=h.device)[None,:]>qpos[:,None]
    for start in range(0,a.num_heads,4):
        logits=(q[:,start:start+4].float()@k[:,start:start+4].float().transpose(-1,-2))/math.sqrt(a.head_dim)
        logits.masked_fill_(future[None,None],float('-inf'))
        links+=torch.softmax(logits,-1).index_select(-1,target)[0].sum(0)
    links/=a.num_heads
    weights=question_attention[qpos].clamp_min(1e-12);weights/=weights.sum()
    weighted=links*weights[:,None]
    score=torch.zeros(n,device=h.device);score[target]=weighted.sum(0)
    partner=torch.full((n,),-1,device=h.device,dtype=torch.long);affinity=torch.zeros(n,device=h.device)
    best,which=weighted.max(0);partner[target]=qpos[which];affinity[target]=best
    best2,which2=links.max(-1);partner[qpos]=target[which2];affinity[qpos]=best2
    return {'cross_score':score,'partner':partner,'affinity':affinity,'link_queries':qpos,'link_targets':target,'link_attention':links,'link_query_weights':weights}
