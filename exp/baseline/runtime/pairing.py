"""Unmodified reference two-question input protocol."""
import hashlib,importlib.util
from pathlib import Path
import torch
from .prompt import question_text, token_ids
CONTEXT='Answer Q1 using CLIP 1 and Q2 using CLIP 2. Each clip has its own audio and timeline. Answer both questions in the same response.'
def assemble(processor,streams,question,context=CONTEXT):
 prompt=question_text(question)
 rebuilt=[];blocks=[];content=[]
 for i,original in enumerate(streams):
  s=dict(original);ids=s['inputs']['input_ids'][0]
  starts=(ids==151652).nonzero().flatten();ends=(ids==151653).nonzero().flatten();assert len(starts)==len(ends)==1
  block=ids[int(starts[0]):int(ends[0])+1];s['av_block']=block
  s['record']=dict(s['record'],question=question['question'])
  ids,_=token_ids(processor,[{'type':'video','video':'clip.mp4'},{'type':'text','text':'Use evidence in this clip if relevant to either question.\n'+prompt}],[block])
  s['inputs']=dict(s['inputs'],input_ids=ids,attention_mask=torch.ones_like(ids))
  rebuilt.append(s);blocks.append(block);content.extend([{'type':'text','text':f'CLIP {i+1}:'},{'type':'video','video':'clip.mp4'}])
 content.append({'type':'text','text':context+'\n'+prompt});ids,text=token_ids(processor,content,blocks)
 audio_masks=[s['inputs']['feature_attention_mask'] for s in rebuilt if 'feature_attention_mask' in s['inputs']]
 width=max((x.shape[-1] for x in audio_masks),default=0)
 masks=torch.cat([torch.nn.functional.pad(x,(0,width-x.shape[-1])) for x in audio_masks]) if audio_masks else torch.zeros((0,0),dtype=torch.long)
 joint={'input_ids':ids,'attention_mask':torch.ones_like(ids),'feature_attention_mask':masks,'video_grid_thw':torch.cat([s['inputs']['video_grid_thw'] for s in rebuilt]),'video_second_per_grid':torch.cat([torch.as_tensor(s['inputs']['video_second_per_grid']).flatten() for s in rebuilt])}
 # No manual context cap: full inputs must be executed or recorded as explicit failures.
 return {'streams':rebuilt,'joint':joint,'context':context,'record':{'item_id':question['id'],'question':question['question'],'frame_count':sum(s['record']['frame_count'] for s in rebuilt)},'prompt_sha256':hashlib.sha256(text.encode()).hexdigest()}
