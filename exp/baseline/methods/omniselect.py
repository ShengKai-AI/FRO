"""OmniSelect official AudioCLIP/NLTK scoring applied to the shared English query translation.
Only feature extraction is microbatched. No translation, gold or ASR is used.
"""
import sys,os
from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace
import torch,numpy as np,torchvision as tv
from PIL import Image
from exp.baseline.runtime.paths import asset,repo
os.environ.setdefault('NLTK_DATA',str(asset('weights')/'nltk_data'))
sys.path.insert(0,str(repo('OmniSelect')/'AudioCLIP'))

def pool_semantic_scores(video_scores,audio_scores,video_indices,audio_indices,video_grid):
 """Align score means to actual interleaved token runs, without changing ranking."""
 def pool(scores,indices):
  chunks=[];start=0
  for i in range(1,len(indices)):
   if indices[i]!=indices[i-1]+1:
    chunks.append(scores[start:i].mean());start=i
  chunks.append(scores[start:].mean())
  return torch.stack(chunks)
 grid_t,grid_h,grid_w=[int(x) for x in video_grid[0]]
 per_grid=grid_h*grid_w//4
 assert len(video_scores)==2*grid_t
 video_per_token=video_scores.reshape(grid_t,2).mean(-1).repeat_interleave(per_grid)
 assert len(video_per_token)==len(video_indices)
 segments=((torch.arange(len(audio_indices),device=audio_scores.device,dtype=torch.float32)+.5)*len(audio_scores)/len(audio_indices)).long().clamp(max=len(audio_scores)-1)
 return pool(video_per_token,video_indices),pool(audio_scores[segments],audio_indices)

class Selector:
 def __init__(self,ratio):
  self.ratio=ratio
  from model import AudioCLIP
  self.net=AudioCLIP(pretrained=str(asset('weights')/'AudioCLIP-Full-Training.pt')).eval().cuda()
  self.transform=tv.transforms.Compose([tv.transforms.ToTensor(),tv.transforms.Resize(224,interpolation=Image.BICUBIC),tv.transforms.CenterCrop(224),tv.transforms.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
 @torch.inference_mode()
 def prepare(self,data,inputs):
  from nltk import word_tokenize,pos_tag
  tokens=word_tokenize(data['record']['question']);tags=pos_tag(tokens)
  keywords=[word for word,tag in tags if tag.startswith('NN') or tag.startswith('JJ')]
  text_input=keywords if keywords else tokens
  n=len(data['visual']);audio=data['audio'];chunk=len(audio)//n
  tracks=torch.from_numpy(audio[:n*chunk]).view(n,1,-1)
  images=torch.stack([self.transform(x) for x in data['visual']])
  a=[];v=[]
  for start in range(0,n,8):
   ((af,_,_),_),_=self.net(audio=tracks[start:start+8].cuda());a.append(af)
   ((_,vf,_),_),_=self.net(image=images[start:start+8].cuda());v.append(vf)
  a=torch.cat(a);v=torch.cat(v)
  # Follow the released scorer's text argument shape, including its keyword joining.
  ((_,_,t),_),_=self.net(text=text_input)
  a=a/a.norm(dim=-1,keepdim=True);v=v/v.norm(dim=-1,keepdim=True);t=t/t.norm(dim=-1,keepdim=True)
  la=(self.net.logit_scale_at.exp().clamp(1,100)*a@t.T).mean(-1)
  lv=(self.net.logit_scale.exp().clamp(1,100)*v@t.T).mean(-1)
  av=la.mean();vv=lv.mean()
  if abs(int(vv)-int(av))<=3.7 or not keywords:av=vv
  regime=True if av<vv else False if av>vv else 'uniform'
  ids=inputs['input_ids'][0];config=self.config
  vl=(ids==config.video_token_id).nonzero(as_tuple=True)[0].tolist();al=(ids==config.audio_token_id).nonzero(as_tuple=True)[0].tolist()
  groups=defaultdict(int)
  for seq in [vl,al]:
   group=1
   for i,index in enumerate(seq):
    if i and index-seq[i-1]!=1:group+=1
    groups[index]=group
  pv,pa=pool_semantic_scores(lv,la,vl,al,inputs['video_grid_thw'])
  return {'pooled_logits_v':pv,'pooled_logits_a':pa,'logits_a':la,'logits_v':lv,'video_first':regime,'args':SimpleNamespace(prune=True,prune_ratio_a=1-self.ratio,prune_ratio_v=1-self.ratio),'token_id2group':groups,'v_lst':vl,'a_lst':al,'visual':data['visual']},{'a_score':float(la.mean()),'v_score':float(lv.mean()),'regime':str(regime),'query_language':'English translation; released English NLTK/AudioCLIP pipeline','keywords':keywords,'semantic_pooling':'actual contiguous modality-token groups; fixes hardcoded four-frame grouping for capped/fractional FPS','video_groups':len(pv),'audio_groups':len(pa)}
