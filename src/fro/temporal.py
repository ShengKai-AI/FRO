"""Five fixed inference-only ideas; no labels, answer access or trainable state."""
import math
import re
import torch
import torch.nn.functional as F
from .rotary import apply_multimodal_rotary_pos_emb, repeat_kv
from .selection import rank01, select_with_coverage


def attach_metadata(parts,data,tokenizer,module,query_mode='all'):
    for part,stream in zip(parts,data['streams']):
        part['source_slot']=int(stream['record']['source'])
        grid=stream['inputs']['video_grid_thw'][0].tolist()
        part['video_shape']=(int(grid[0]),int(grid[1])//2,int(grid[2])//2)
        if module not in ('query_contrast','option_evidence') and query_mode=='all': continue
        ids=part['original_ids'].tolist()
        start=max(i for i,t in enumerate(ids) if t in (151646,151656))+1
        tail=ids[start:]
        text=tokenizer.decode(tail,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        m1=text.index('Q1 (CLIP 1): ');a=m1+len('Q1 (CLIP 1): ')
        m2=text.index('Q2 (CLIP 2): ');b=m2+len('Q2 (CLIP 2): ')
        end=text.index('<|im_end|>',b)
        stops=[0]+[len(tokenizer.decode(tail[:i+1],skip_special_tokens=False,clean_up_tokenization_spaces=False)) for i in range(len(tail))]
        def span(lo,hi):
            ix=[start+j for j in range(len(tail)) if stops[j]<hi and stops[j+1]>lo and tail[j]<151643]
            assert ix,(lo,hi,text)
            return torch.tensor(ix,device=part['embeds'].device)
        part['content_queries']=torch.cat((span(a,m2),span(b,end)))
        ranges=((a,m2),(b,end))
        own_lo,own_hi=ranges[part['source_slot']-1]
        if query_mode=='own': part['selected_queries']=span(own_lo,own_hi)
        elif query_mode=='both_content': part['selected_queries']=part['content_queries']
        elif query_mode in ('own_stem','both_stems','own_options'):
            stems=[];options=[]
            for lo,hi in ranges:
                label=re.search(r'(?m)^A\. ',text[lo:hi])
                assert label is not None
                cut=lo+label.start()
                stems.append(span(lo,cut));options.append(span(cut,hi))
            if query_mode=='both_stems': part['selected_queries']=torch.cat(stems)
            elif query_mode=='own_stem': part['selected_queries']=stems[part['source_slot']-1]
            else: part['selected_queries']=options[part['source_slot']-1]
        prefix=text.index('Use evidence in this clip')
        part['generic_queries']=span(prefix,m1)
        if module=='option_evidence':
            lo,hi=(a,m2) if part['source_slot']==1 else (b,end)
            block=text[lo:hi]
            labels=list(re.finditer(r'(?m)^([ABCD])\. ',block))
            assert [m.group(1) for m in labels]==list('ABCD')
            part['option_queries']=[span(lo+m.end(),lo+labels[j+1].start() if j<3 else hi) for j,m in enumerate(labels)]


@torch.inference_mode()
def head_attention(layer,h,pe,qpos,max_queries=32,need_value=False,quality_groups=None):
    """Mean causal question attention per head; no N*N map or decoder mutation."""
    if len(qpos)>max_queries:
        qpos=qpos[torch.linspace(0,len(qpos)-1,max_queries,device=h.device).round().long()]
    assert len(qpos)
    attn=layer.self_attn;n=h.shape[1];norm=layer.input_layernorm(h)
    q=attn.q_proj(norm[:,qpos]).view(1,len(qpos),attn.num_heads,attn.head_dim).transpose(1,2)
    k=attn.k_proj(norm).view(1,n,attn.num_key_value_heads,attn.head_dim).transpose(1,2)
    cos,sin=pe
    q,_=apply_multimodal_rotary_pos_emb(q,q,cos[:,:,qpos],sin[:,:,qpos],attn.rope_scaling['mrope_section'])
    k,_=apply_multimodal_rotary_pos_emb(k,k,cos,sin,attn.rope_scaling['mrope_section'])
    k=repeat_kv(k,attn.num_key_value_groups)
    heads=[];qualities=[]
    future=torch.arange(n,device=h.device)[None,:]>qpos[:,None]
    for first in range(0,attn.num_heads,4):
        logits=(q[:,first:first+4].float() @ k[:,first:first+4].float().transpose(-1,-2))/math.sqrt(attn.head_dim)
        logits.masked_fill_(future[None,None],-torch.inf)
        probabilities=torch.softmax(logits,-1)
        heads.append(probabilities.mean((0,2)))
        if quality_groups is not None:
            qualities.append(torch.stack([paq_quality(probabilities[0,:,:,quality_groups==mod]) for mod in (0,1)],-1))
    per_head=torch.cat(heads)
    value_norm=None
    if need_value:
        v=attn.v_proj(norm).view(1,n,attn.num_key_value_heads,attn.head_dim).transpose(1,2)
        # Repeat scalar norms rather than a full expanded value tensor.
        value_norm=v[0].float().abs().sum(-1).repeat_interleave(attn.num_key_value_groups,dim=0)
    if quality_groups is not None:return per_head,qpos,value_norm,torch.cat(qualities)
    return per_head,qpos,value_norm


def neutral_rank(x):
    # No invented ordering when a new diagnostic contains no signal.
    if len(x)<=1 or float(x.max()-x.min())<=1e-12:
        return torch.full_like(x,.5,dtype=torch.float32)
    _,inverse,counts=torch.unique(x,sorted=True,return_inverse=True,return_counts=True)
    lo=counts.cumsum(0)-counts
    mid=(lo.float()+(counts.float()-1)/2)/max(1,len(x)-1)
    return mid[inverse]


def contrast_scores(content,generic,groups):
    """Log modality-normalized likelihood ratio against generic instructions."""
    score=torch.zeros_like(content)
    for mod in (0,1):
        mask=groups==mod;n=int(mask.sum())
        p=content[mask].clamp_min(0);q=generic[mask].clamp_min(0)
        p=p/p.sum().clamp_min(1e-12);q=q/q.sum().clamp_min(1e-12)
        eps=.01/n
        score[mask]=torch.log(p+eps)-torch.log(q+eps)
    return score


def head_selected_scores(per_head,groups,head_count=8):
    """Pick heads by attention mass in each modality; normalize within modality."""
    score=torch.zeros(per_head.shape[-1],device=per_head.device);selected={}
    for mod in (0,1):
        mask=groups==mod
        mass=per_head[:,mask].sum(-1)
        ix=torch.argsort(mass,descending=True,stable=True)[:min(head_count,len(mass))]
        normalized=per_head[ix][:,mask]/mass[ix,None].clamp_min(1e-12)
        score[mask]=normalized.mean(0)
        selected[str(mod)]=ix.cpu().tolist()
    return score,selected


def value_scores(per_head,value_norm):
    return (per_head*value_norm).mean(0)


def option_dispersion(option_maps,groups):
    """Choice-normalized dispersion; high only where option maps differ."""
    score=torch.zeros(option_maps.shape[-1],device=option_maps.device)
    for mod in (0,1):
        mask=groups==mod;x=option_maps[:,mask]
        x=x/x.sum(-1,keepdim=True).clamp_min(1e-12)
        score[mask]=x.std(dim=0,unbiased=False)
    return score


def temporal_novelty(original,groups,video_shape):
    """Encoder feature change, not optical flow or a certified action detector."""
    out=torch.zeros(len(groups),device=original.device)
    for mod in (0,1):
        ii=(groups==mod).nonzero().flatten()
        if not len(ii): continue
        x=F.normalize(original[ii].float(),dim=-1)
        if mod==1:
            t,h,w=video_shape;assert len(ii)==t*h*w
            x=x.view(t,h*w,-1)
        else: x=x[:,None,:]
        delta=(1-(x[1:]*x[:-1]).sum(-1)).clamp(0,2)
        total=torch.zeros(x.shape[:-1],device=x.device);count=torch.zeros_like(total)
        total[1:]+=delta;total[:-1]+=delta;count[1:]+=1;count[:-1]+=1
        out[ii]=(total/count.clamp_min(1)).flatten()
    return out


def select_with_novelty(score,relevance,novelty,k,fraction=.2,coverage=.2):
    """Reserve <=20% budget for high-change query-relevant tokens; exact total."""
    ordinary=select_with_coverage(score,k,coverage)
    if float(novelty.max()-novelty.min())<=1e-8:
        return ordinary,0
    need=max(1,min(k,round(k*fraction)))
    utility=neutral_rank(novelty)*(.5+.5*relevance)
    mandatory=torch.zeros_like(ordinary)
    mandatory[torch.topk(utility,need).indices]=True
    # Keep the original coverage representatives as well where budget permits.
    cover=select_with_coverage(score,max(1,round(k*coverage)),1.)
    selected=mandatory.clone()
    extras=(cover&~selected).nonzero().flatten();room=k-int(selected.sum())
    if len(extras)>room:
        extras=extras[torch.topk(score[extras],room).indices] if room else extras[:0]
    selected[extras]=True;room=k-int(selected.sum())
    if room:selected[torch.topk(score.masked_fill(selected,-torch.inf),room).indices]=True
    assert int(selected.sum())==k and bool(selected[mandatory].all())
    return selected,int((selected&~ordinary).sum())


def diversity_penalty(score, features, strength=.1):
    """Penalize redundancy to stronger representatives from <=64 temporal bins.

    Bins follow encoder order. This approximates diversity selection without
    an N*N map or an O(K) full-dimensional MMR loop for thousands of survivors.
    """
    n=len(score); bins=min(64,n)
    group=torch.arange(n,device=score.device)*bins//n
    best=torch.full((bins,),-torch.inf,device=score.device)
    best.scatter_reduce_(0,group,score,reduce='amax',include_self=True)
    candidate=torch.where(score==best[group],torch.arange(n,device=score.device),n)
    reps=torch.full((bins,),n,device=score.device,dtype=torch.long)
    reps.scatter_reduce_(0,group,candidate,reduce='amin',include_self=True)
    sim=(features @ features[reps].T).float().clamp(0,1)
    index=torch.arange(n,device=score.device)
    stronger=(score[reps][None] > score[:,None]) | ((score[reps][None]==score[:,None]) & (reps[None]<index[:,None]))
    redundancy=sim.masked_fill(~stronger,0).max(-1).values
    return score-strength*redundancy



def temporal_smooth(score,groups,video_shape,mix):
    """Convex adjacent-time smoothing; video never crosses spatial patches."""
    out=score.clone()
    for mod in (0,1):
        ii=(groups==mod).nonzero().flatten()
        if mod==1:
            t,h,w=video_shape
            assert len(ii)==t*h*w
            x=score[ii].view(t,h*w)
        else: x=score[ii,None]
        total=x.clone();count=torch.ones_like(x)
        total[1:]+=x[:-1];count[1:]+=1
        total[:-1]+=x[1:];count[:-1]+=1
        out[ii]=((1-mix)*x+mix*total/count).flatten()
    return out


def paq_quality(attention):
    """I(query;token)/H(token), independently normalized within each modality."""
    p=attention.float()/attention.float().sum(-1,keepdim=True).clamp_min(1e-12)
    marginal=p.mean(-2)
    entropy=-(marginal*marginal.clamp_min(1e-12).log()).sum(-1)
    conditional=-(p*p.clamp_min(1e-12).log()).sum(-1).mean(-1)
    return ((entropy-conditional).clamp_min(0)/entropy.clamp_min(1e-12)).clamp(0,1)


def quality_aggregate(heads,groups,mode,quality=None,temperature=1.,boost=.5,value_norm=None):
    score=torch.zeros(heads.shape[-1],device=heads.device);weights={}
    for mod in (0,1):
        mask=groups==mod;a=heads[:,mask]
        contribution=a if value_norm is None else a*value_norm[:,mask]
        if mode=='specialization':
            distribution=a/a.sum(0,keepdim=True).clamp_min(1e-12)
            entropy=-(distribution*distribution.clamp_min(1e-12).log()).sum(0)/math.log(max(2,len(a)))
            score[mask]=contribution.mean(0)*(1+boost*(1-entropy).clamp(0,1))
            weights[str(mod)]='token-wise head-specialization factor'
        else:
            if mode=='paq':q=quality[:,mod]
            elif mode=='concentration':
                p=a/a.sum(-1,keepdim=True).clamp_min(1e-12)
                q=1+(p*p.clamp_min(1e-12).log()).sum(-1)/math.log(max(2,a.shape[-1]))
            else:raise ValueError(mode)
            w=torch.softmax(q/temperature,dim=0)
            score[mask]=(contribution*w[:,None]).sum(0)
            weights[str(mod)]=w.cpu().tolist()
    return score,weights
