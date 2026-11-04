import torch
from .video import slice_part

def compress(part, thinker, ratio, seed):
    ids=part['ids']; keep=torch.ones_like(ids,dtype=torch.bool)
    generator=torch.Generator(device=ids.device).manual_seed(seed)
    for token in (thinker.config.audio_token_id,thinker.config.video_token_id):
        slots=(ids==token).nonzero().flatten()
        if not len(slots): continue
        keep[slots]=False
        chosen=torch.randperm(len(slots),generator=generator,device=ids.device)[:max(1,round(len(slots)*ratio))]
        keep[slots[chosen]]=True
    return slice_part(part,keep)
