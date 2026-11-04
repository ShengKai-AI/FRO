"""DyCoke temporal ranking with exact integer budgets in four-frame groups.

AV adaptation: each group retains its first frame; non-anchor frames keep their
least-similar spatial tokens. Allocate integer remainders in temporal order.
This is TTM only, with no decode-time KV compression. Counts below the anchor
floor are explicitly unattainable. No post-hoc padding/truncation is used.
"""
import torch
import torch.nn.functional as F

def compress(features,frames,tokens_per_frame,ratio):
    x=features.reshape(frames,tokens_per_frame,-1)
    target=round(frames*tokens_per_frame*ratio)
    anchors=list(range(0,frames,4));floor=len(anchors)*tokens_per_frame
    if target<floor:raise ValueError(f'DyCoke anchor floor {floor} exceeds requested {target}')
    candidates=[i for i in range(frames) if i not in anchors]
    remaining=target-floor
    base,remainder=divmod(remaining,len(candidates)) if candidates else (0,0)
    selected=[torch.arange(i*tokens_per_frame,(i+1)*tokens_per_frame,device=x.device) for i in anchors]
    for slot,i in enumerate(candidates):
        count=base+(slot<remainder)
        # Odd frames use their preceding even frame; frame 2 uses frame 0.
        reference=i-1 if i%2 else i-2
        similarities=F.cosine_similarity(x[i].float(),x[reference].float(),dim=-1)
        indices=torch.argsort(similarities,stable=True)[:count]+i*tokens_per_frame
        selected.append(indices)
    indices=torch.cat(selected).sort().values
    if len(indices)!=target:raise ValueError('Unattainable integer budget')
    return features[indices],indices
