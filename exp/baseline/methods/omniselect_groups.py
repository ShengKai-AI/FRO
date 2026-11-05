"""OmniSelect ranking with exact modality-local contiguous-group indexing.

The released traversal drops the last feature from slices and assumes matching
A/V group transitions. Shared frame caps and timestamp-preserving audio sampling
require grouping each modality independently. AudioCLIP routing and native
weight allocation/rankings are retained.
"""
import torch

@torch.no_grad()
def omniselect(video_grid_thw,input_embeds,audio_features,vision_features,
               attn_mean_a,attn_mean_v,input_ids,prune_need):
    from OmniSelect import allocate_pruning_weights
    ids=input_ids.flatten();keep=torch.ones_like(ids,dtype=torch.bool)
    for kind,slots,features,pooled in [
        ('video',prune_need['v_lst'],attn_mean_v,prune_need['pooled_logits_v']),
        ('audio',prune_need['a_lst'],attn_mean_a,prune_need['pooled_logits_a'])]:
        if not slots:continue
        if len(features)!=len(slots):raise ValueError(f'{kind} token/feature length mismatch: {len(slots)} vs {len(features)}')
        boundaries=[0]+[i for i in range(1,len(slots)) if slots[i]!=slots[i-1]+1]+[len(slots)]
        sizes=[b-a for a,b in zip(boundaries,boundaries[1:])]
        if len(pooled)!=len(sizes):raise ValueError('Semantic group count mismatch')
        ratio=getattr(prune_need['args'],'prune_ratio_v' if kind=='video' else 'prune_ratio_a')
        routed=prune_need['video_first'] is (kind=='video')
        weights=allocate_pruning_weights(pooled,sizes,ratio) if routed and len(sizes)>1 else torch.full_like(pooled,ratio)
        # Round the total modality budget once, then distribute remainders.
        # Per-group flooring otherwise creates large simultaneous count jumps.
        capacities=torch.tensor(sizes,device=weights.device,dtype=torch.long)
        raw=capacities.double()*(1-weights.double())
        counts=raw.floor().long().clamp(min=1);counts=torch.minimum(counts,capacities)
        total=max(len(sizes),min(sum(sizes),round(float(raw.sum()))))
        delta=total-int(counts.sum())
        priority=torch.argsort(raw-counts.double(),descending=delta>0,stable=True).tolist()
        while delta:
            changed=False
            for slot in priority:
                if delta>0 and counts[slot]<capacities[slot]:counts[slot]+=1;delta-=1;changed=True
                elif delta<0 and counts[slot]>1:counts[slot]-=1;delta+=1;changed=True
                if not delta:break
            if not changed:raise ValueError('No feasible integer modality budget')
        for index,(start,end) in enumerate(zip(boundaries,boundaries[1:])):
            values=features[start:end]
            scores=-(values@values.T).mean(0) if kind=='video' else values.flatten()
            n=end-start;k=int(counts[index])
            chosen=torch.topk(scores,k,largest=True,sorted=False).indices
            local=torch.zeros(n,dtype=torch.bool,device=ids.device);local[chosen]=True
            keep[torch.as_tensor(slots[start:end],device=ids.device)]=local
    return keep
