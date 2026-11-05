from pathlib import Path
import torch,hashlib
SYSTEM={'role':'system','content':[{'type':'text','text':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'}]}
def question_text(q):
 if q.get('protocol')!='native_two_video_mc' or len(q['questions'])!=1:raise ValueError('Expected one native question')
 x=q['questions'][0]
 return 'Use both videos to answer the single question. Output only one option letter. Do not add explanations.\n'+x['question']+'\n'+'\n'.join(x['options'])
def token_ids(processor,content,blocks):
 text=processor.apply_chat_template([SYSTEM,{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
 ids=processor.tokenizer(text,return_tensors='pt')['input_ids'][0];starts=(ids==151652).nonzero().flatten().tolist();ends=(ids==151653).nonzero().flatten().tolist();assert len(starts)==len(ends)==len(blocks)
 parts=[];offset=0
 for lo,hi,block in zip(starts,ends,blocks):parts.extend([ids[offset:lo],block]);offset=hi+1
 parts.append(ids[offset:]);result=torch.cat(parts)[None];return result,text
