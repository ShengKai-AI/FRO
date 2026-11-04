"""Per-stream video compressors; shared deterministic audio policy; no gold access."""
import sys,math,copy
from pathlib import Path
import torch
from exp.baseline.runtime.paths import repo
sys.path.append(str(repo('VidCom2-omni')))

def install_encoder_stats(thinker):
 from baselines.visionzip import (Qwen2_5OmniVisionFlashAttention2,Qwen2_5OmniVisionBlock,Qwen2_5OmniVisionEncoder,
 Qwen2_5OmniVisionFlashAttention2_forward_visionzip,Qwen2_5OmniVisionBlock_forward_visionzip,Qwen2_5OmniVisionEncoder_forward_visionzip)
 Qwen2_5OmniVisionFlashAttention2.forward=Qwen2_5OmniVisionFlashAttention2_forward_visionzip
 Qwen2_5OmniVisionBlock.forward=Qwen2_5OmniVisionBlock_forward_visionzip
 Qwen2_5OmniVisionEncoder.forward=Qwen2_5OmniVisionEncoder_forward_visionzip
 thinker.visual.grid_in_window=2

def uniform(n,k,device):
 return torch.linspace(0,n-1,k,device=device).round().long().unique(sorted=True)

def slice_part(part,keep,emb=None):
 out=dict(part);out['embeds']=(part['embeds'] if emb is None else emb)[:,keep];out['positions']=part['positions'][...,keep];out['ids']=part['ids'][keep]
 final=part['final'].clone();final[part['final']]=keep.cpu();out['final']=out['stage1']=final
 assert torch.isfinite(out['embeds']).all()
 return out

def common_audio(part,raw,aid,ratio):
 # Restore audio from raw, preserving selected video and original positions.
 keep=part['final'].to(raw['ids'].device).clone();am=raw['ids']==aid;slots=am.nonzero().flatten();keep[slots]=False
 chosen=slots[uniform(len(slots),min(len(slots),max(1,round(ratio*len(slots)))),slots.device)];keep[chosen]=True
 emb=raw['embeds'].clone();old=part['final'].to(emb.device);emb[:,old]=part['embeds']
 out=slice_part(raw,keep,emb)
 out['score_diagnostics']=part.get('score_diagnostics',{})
 return out

@torch.inference_mode()
def apply_video(part,stream,thinker,method,ratio,seed):
 aid,vid=thinker.config.audio_token_id,thinker.config.video_token_id
 ids=part['ids'];vs=(ids==vid).nonzero().flatten();grid=stream['inputs']['video_grid_thw'][0].to(ids.device)
 feat=part['embeds'][0,vs];n=len(vs);t,h,w=map(int,grid.tolist());tpf=h*w//4
 assert n==t*tpf and n>0
 attention=part.get('vision_attention');keys=part.get('vision_keys');calls=0
 def algorithm(r):
  nonlocal calls
  calls+=1
  # Fixed, item-local selector RNG: budget calibration must not tune random draws.
  torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  if method=='uniform':
   idx=uniform(n,max(1,int(r*n)),feat.device);return feat[idx],idx
  if method=='vidcom2':
   from token_compressor.vidcom2.vidcom2 import select_low_var_channels,compute_gaussian_scores,compute_scales,select_outlier_indices,_map_linear_offset
   a,b=compute_gaussian_scores(select_low_var_channels(feat),tpf)
   idx=_map_linear_offset(select_outlier_indices(a+b,compute_scales(-a.mean(-1),r),tpf),tpf)
   return feat[idx],idx
  if method=='fastvid':
   from .fastvid_core import compress
   return compress(feat,grid,attention,r)
  if method=='dycoke_ttm':
   from .dycoke_ttm import compress
   return compress(feat,t,tpf,r)
  if method=='visionzip':
   from baselines.visionzip.visionzip_units import video_visionzip
   # 90% dominant + 10% contextual within the retained budget.
   emb,mask=video_visionzip(part['embeds'].clone(),ids[None],attention,keys,vid,r,r*.1,2,4,grid[None])
   idx=mask[vs].nonzero().flatten();return emb[0,vs[idx]],idx
  raise ValueError(method)
 features,indices=algorithm(ratio);native_ratio=len(indices)/n;effective=ratio
 # Find nearest attainable native output count, without editing ranking or merged outputs.
 if abs(native_ratio-ratio)>.005 and method not in ('uniform',):
  best=(abs(native_ratio-ratio),ratio,features,indices);lo,hi=0.001,.999
  for _ in range(10):
   r=(lo+hi)/2;f,k=algorithm(r);observed=len(k)/n
   if abs(observed-ratio)<best[0]:best=(abs(observed-ratio),r,f,k)
   if observed<ratio:lo=r
   else:hi=r
   if best[0]<=.005:break
  _,effective,features,indices=best
 assert len(indices)==len(indices.unique()) and len(indices)==len(features)
 order=indices.argsort();indices=indices[order];features=features[order]
 keep=torch.ones_like(ids,dtype=torch.bool);keep[vs]=False;keep[vs[indices]]=True
 emb=part['embeds'].clone();emb[0,vs[indices]]=features
 out=common_audio(slice_part(part,keep,emb),part,aid,ratio)
 out['aux']={'native_video_ratio_at_target_parameter':native_ratio,'effective_parameter':effective,'calibration_calls':calls,'actual_video_ratio':len(indices)/n,'budget_matched':abs(len(indices)/n-ratio)<=.005,'selector_seed':seed,'audio_policy':'uniform', 'target_ratio':ratio,'variant':method}
 for key in ['vision_attention','vision_keys']:out.pop(key,None)
 return out
